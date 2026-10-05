#!/usr/bin/env python
"""LLM-as-Judge re-evaluation for the korean_character_ocr task.

The built-in OCR metrics (ocr_strict_em / ocr_format_valid) reject any
response that does not follow an exact "OCR: xxx" format, so a model that
reads the text correctly but wraps it differently scores 0. This script
instead asks Gemini whether each prediction captured the gold text by
*content*, ignoring formatting, word order, and extra text.

It reads per-sample predictions straight from lighteval `--save-details`
parquet files — there is NO model re-inference, only judge API calls.

Cache: ocr_llm_judge_cache.json. Cache keys embed a hash of JUDGE_PROMPT,
so editing the prompt automatically invalidates stale entries.

Usage:
    python scripts/ocr_llm_judge.py
    python scripts/ocr_llm_judge.py --workers 8 --show-reasoning
"""
import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import time

import pyarrow.parquet as pq
from dotenv import dotenv_values
from google import genai
from google.genai import types

ENV_FILE = "/workspace/llm_web_translation_mt/.env"
CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "ocr_llm_judge_cache.json")
GEMINI_MODEL = "gemini-3-flash-preview"

# Model -> directory holding korean_character_ocr `details` parquet files.
# NOTE: update these paths if the result directories are reorganized.
MODELS = {
    "qwen3.5-9b (base)": "test_final_11_tasks/Qwen_Qwen3.5-9B/korean_character_ocr/details",
    "dpo ckpt-1253": "test_final_11_tasks/local_qwen3.5_9b_dpo/korean_character_ocr/details",
    "dpo ckpt-600": "test_final_11_tasks/local_models_qwen3_5_9b_multimodal_sft_dpo_normalized_checkpoint-600_standalone/korean_character_ocr/details",
    "sft lr1e-5 ckpt-300": "test_final_11_tasks/local_models_qwen3_5_9b_multimodal_sft_lr_1e-5_checkpoint-300_standalone/korean_character_ocr/details",
    "mammothvl2": "test_final_11_tasks_endpoint/qwen3_5_9b_mammothvl2_sft/korean_character_ocr/details",
    "mammothvl2_korean": "test_final_11_tasks_endpoint/qwen3_5_9b_mammothvl2_korean_sft/korean_character_ocr/details",
}

JUDGE_PROMPT = """당신은 한국어 OCR 평가자입니다. 이미지 속 텍스트를 모델이 올바르게 읽었는지 판정합니다.

[정답 텍스트]
{gold}

[모델 응답]
{pred}

[판정 기준] — 아래는 모두 무시하고 글자 내용만 봅니다:
- 띄어쓰기 / 문장부호(· , . " ' 등) 차이
- "사진 속 텍스트:", "~라고 되어 있습니다" 같은 포맷·설명 표현
- 단어·구절의 순서 차이
- 모델이 정답 외에 추가로 더 읽은 텍스트(예: 전화번호)

판정:
- Yes: 정답 텍스트의 모든 핵심 글자·단어가 모델 응답에 들어 있음 (순서·추가내용 무관)
- No: 정답 텍스트의 일부가 누락되었거나, 다른 글자로 잘못 읽음

주의: 정답 텍스트 앞뒤에 불필요한 기호 조각(따옴표, "에 " 등)이 섞여 있을 수 있습니다.
그런 noise 는 무시하고 실제 간판·문구 내용만 기준으로 삼으세요.

[예시]
정답: 클리닉독서실 | 응답: "클리닉 독서실"이라고 되어 있습니다.
-> {{"reasoning":"띄어쓰기 차이뿐, 핵심 글자 모두 일치","verdict":"Yes"}}
정답: 삼겹살 백반 말죽거리 | 응답: 삼겹살·백반 말죽거리 548-9921
-> {{"reasoning":"정답 단어 모두 포함, 전화번호는 추가로 더 읽은 것이라 무관","verdict":"Yes"}}
정답: 풀하우스 악세사리 잡화 | 응답: 풀하우스
-> {{"reasoning":"'악세사리 잡화' 누락","verdict":"No"}}

먼저 reasoning, 그 다음 verdict 를 정해 JSON 으로만 답하세요:
{{"reasoning":"<짧은 근거>","verdict":"Yes" 또는 "No"}}"""

PROMPT_VER = hashlib.md5(JUDGE_PROMPT.encode()).hexdigest()[:8]

_CURVED = "‘’“”"  # ' ' " "


def clean_gold(gold: str) -> str:
    """Repair gold_text fragments left by the (now-fixed) extraction bug.

    Existing parquet files still hold golds like "에 '올댓발레"; this strips
    the leftover "에 " + quote prefix and any wrapping quotes.
    """
    g = gold.strip()
    g = re.sub(rf"^에\s*['\"{_CURVED}]\s*", "", g)
    g = g.strip(" \t'\"" + _CURVED)
    return g.strip()


def latest_parquet(base_dir: str) -> str | None:
    files = sorted(glob.glob(os.path.join(base_dir, "**", "details_*.parquet"), recursive=True))
    return files[-1] if files else None


def judge(client: genai.Client, gold: str, pred: str) -> dict:
    """Return {'verdict': 'Yes'|'No'|'?', 'reasoning': str}."""
    prompt = JUDGE_PROMPT.format(gold=gold, pred=pred)
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    temperature=0.0,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
            )
            raw = (resp.text or "").strip()
            obj = json.loads(raw)
            verdict = str(obj.get("verdict", "")).strip()
            if verdict.lower().startswith("yes"):
                verdict = "Yes"
            elif verdict.lower().startswith("no"):
                verdict = "No"
            else:
                verdict = "?"
            return {"verdict": verdict, "reasoning": str(obj.get("reasoning", ""))}
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"verdict": "?", "reasoning": f"ERROR:{type(e).__name__}:{e}"}
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--show-reasoning", action="store_true", help="print per-sample reasoning")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    key = dotenv_values(ENV_FILE)["GEMINI_API_KEY"].strip('"')
    client = genai.Client(api_key=key)

    cache = {}
    if os.path.exists(CACHE_FILE):
        cache = json.load(open(CACHE_FILE))

    # Build task list
    tasks = []
    for mname, base in MODELS.items():
        p = latest_parquet(base)
        if not p:
            print(f"  WARNING {mname}: no parquet under {base}")
            continue
        t = pq.read_table(p).to_pandas()
        print(f"  loaded {mname}: {len(t)} rows")
        for i, row in t.iterrows():
            gold = clean_gold(row["doc"]["specific"]["gold_text"])
            pred = row["model_response"]["text"]
            # endpoint 평가 parquet은 text가 ndarray/list로 저장됨 → 첫 응답 추출
            if not isinstance(pred, str):
                pred = (pred[0] if len(pred) else "") if hasattr(pred, "__len__") else str(pred)
            digest = hashlib.md5((gold + "|||" + pred).encode()).hexdigest()
            ck = f"{PROMPT_VER}|{mname}|{i}|{digest}"
            tasks.append((ck, mname, i, gold, pred))

    todo = [t for t in tasks if t[0] not in cache]
    print(f"\n total={len(tasks)} cached={len(tasks) - len(todo)} todo={len(todo)} (prompt_ver={PROMPT_VER})\n")

    if todo:
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut = {ex.submit(judge, client, t[3], t[4]): t for t in todo}
            done = 0
            for f in concurrent.futures.as_completed(fut):
                ck = fut[f][0]
                cache[ck] = f.result()
                done += 1
                if done % 40 == 0:
                    json.dump(cache, open(CACHE_FILE, "w"), ensure_ascii=False, indent=1)
                    print(f"  ... {done}/{len(todo)} ({time.time() - t0:.0f}s)")
        json.dump(cache, open(CACHE_FILE, "w"), ensure_ascii=False, indent=1)
        print(f" judged in {time.time() - t0:.0f}s\n")

    # Aggregate
    print(f"{'Model':25} | {'Yes':>4} | {'No':>4} | {'?':>3} | {'Yes ratio':>9}")
    print("-" * 62)
    for mname in MODELS:
        yes = no = err = 0
        for ck, m, i, gold, pred in tasks:
            if m != mname:
                continue
            v = cache.get(ck, {}).get("verdict", "?")
            if v == "Yes":
                yes += 1
            elif v == "No":
                no += 1
            else:
                err += 1
        total = yes + no + err
        ratio = yes / total if total else 0.0
        print(f"{mname:25} | {yes:>4} | {no:>4} | {err:>3} | {ratio:>8.1%}")

    if args.show_reasoning:
        print("\n=== per-sample reasoning ===")
        for ck, m, i, gold, pred in tasks:
            entry = cache.get(ck, {})
            print(f"[{m}] s{i} {entry.get('verdict','?')}: gold={gold!r} pred={pred[:50]!r}")
            print(f"     reason: {entry.get('reasoning','')}")


if __name__ == "__main__":
    main()
