#!/usr/bin/env python
"""LLM-as-Judge re-evaluation for the korean_heritage_name_vqa task.

The built-in name_vqa metrics (name_vqa_relaxed_em / strict_em) demand an
exact "Answer: <name>" format and near-exact string match, so a response
that names the correct heritage item in a different phrasing scores 0.
This script instead asks Gemini whether the prediction refers to the SAME
heritage item as the gold answer, under a lenient policy: if it points to
the same item it counts, even without the exact official name.

Reads per-sample predictions from lighteval `--save-details` parquet
files — there is NO model re-inference, only judge API calls.

Cache: name_vqa_llm_judge_cache.json. Cache keys embed a hash of
JUDGE_PROMPT, so editing the prompt auto-invalidates stale entries.

Usage:
    python scripts/name_vqa_llm_judge.py
    python scripts/name_vqa_llm_judge.py --workers 8 --show-reasoning
"""
import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import time

import pyarrow.parquet as pq
from dotenv import dotenv_values
from google import genai
from google.genai import types

ENV_FILE = "/workspace/llm_web_translation_mt/.env"
CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "name_vqa_llm_judge_cache.json")
GEMINI_MODEL = "gemini-3-flash-preview"

# Model -> directory holding korean_heritage_name_vqa `details` parquet files.
# NOTE: update these paths if the result directories are reorganized.
MODELS = {
    "qwen3.5-9b (base)": "test_final_11_tasks/Qwen_Qwen3.5-9B/korean_heritage_name_vqa/details",
    "dpo ckpt-1253": "test_final_11_tasks/local_qwen3.5_9b_dpo/korean_heritage_name_vqa/details",
    "dpo ckpt-600": "test_final_11_tasks/local_models_qwen3_5_9b_multimodal_sft_dpo_normalized_checkpoint-600_standalone/korean_heritage_name_vqa/details",
    "sft lr1e-5 ckpt-300": "test_final_11_tasks/local_models_qwen3_5_9b_multimodal_sft_lr_1e-5_checkpoint-300_standalone/korean_heritage_name_vqa/details",
    "dpo ckpt-600 (vllm dp8)": "test_final_11_tasks_endpoint_vllm_dp8/qwen3_5_9b_multimodal_sft_dpo_normalized_checkpoint-600/korean_heritage_name_vqa/details",
    "simple_name_sft (vllm dp8)": "test_final_11_tasks_endpoint_vllm_dp8/qwen3_5_9b_simple_name_sft/korean_heritage_name_vqa/details",
}

JUDGE_PROMPT = """당신은 한국 문화재 식별 평가자입니다. 사진 속 문화재의 이름을 모델이 올바르게 답했는지 판정합니다.

[정답 문화재명]
{gold}

[모델 응답]
{pred}

[판정 기준] — 관대 기준으로 판단합니다:
- 모델 응답이 정답과 **같은 문화재**를 가리키면 Yes 입니다.
- 정확한 공식 명칭이 아니어도, 핵심 명칭으로 같은 문화재를 특정할 수 있으면 Yes.
- "Answer:", "이 사진은 ~입니다", "사진 속 텍스트:" 같은 포맷·설명 표현은 무시하고 가리키는 대상만 봅니다.
- 다른 문화재를 가리키거나, 문화재를 특정할 수 없을 만큼 모호하면(예: 등급·번호만 답함, 사진 속 글자만 옮겨 적음) No.

[예시]
정답: 경복궁 근정전 | 응답: 이 사진은 경복궁의 근정전입니다.
-> {{"reasoning":"포맷만 다를 뿐 같은 문화재를 정확히 지칭","verdict":"Yes"}}
정답: 구례 화엄사 각황전 앞 석등 | 응답: Answer: 화엄사 석등
-> {{"reasoning":"화엄사의 석등으로 같은 대상을 가리킴 (관대 기준)","verdict":"Yes"}}
정답: 제천 입석리 선돌 | 응답: 안동석굴사
-> {{"reasoning":"완전히 다른 문화재","verdict":"No"}}
정답: 구례 화엄사 각황전 앞 석등 | 응답: 보물 제1056호
-> {{"reasoning":"문화재 등급·번호만 답해 어떤 문화재인지 특정 불가","verdict":"No"}}

먼저 reasoning, 그 다음 verdict 를 정해 JSON 으로만 답하세요:
{{"reasoning":"<짧은 근거>","verdict":"Yes" 또는 "No"}}"""

PROMPT_VER = hashlib.md5(JUDGE_PROMPT.encode()).hexdigest()[:8]


def latest_parquet(base_dir: str) -> str | None:
    files = sorted(glob.glob(os.path.join(base_dir, "**", "details_*.parquet"), recursive=True))
    return files[-1] if files else None


def get_gold(row) -> str:
    spec = row["doc"]["specific"]
    if isinstance(spec, dict) and spec.get("answer"):
        return str(spec["answer"]).strip()
    choices = row["doc"]["choices"]
    if choices is not None and len(choices):
        return str(choices[row["doc"]["gold_index"]]).strip()
    return ""


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
            obj = json.loads((resp.text or "").strip())
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
    parser.add_argument("--show-reasoning", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    key = dotenv_values(ENV_FILE)["GEMINI_API_KEY"].strip('"')
    client = genai.Client(api_key=key)

    cache = {}
    if os.path.exists(CACHE_FILE):
        cache = json.load(open(CACHE_FILE))

    tasks = []
    for mname, base in MODELS.items():
        p = latest_parquet(base)
        if not p:
            print(f"  WARNING {mname}: no parquet under {base}")
            continue
        t = pq.read_table(p).to_pandas()
        print(f"  loaded {mname}: {len(t)} rows")
        for i, row in t.iterrows():
            gold = get_gold(row)
            pred_text = row["model_response"]["text"]
            # `text` may be a scalar str (accelerate backend) or a list/ndarray
            # of completions (endpoint litellm backend). Normalize to a str.
            if not isinstance(pred_text, str):
                try:
                    pred_text = pred_text[0] if len(pred_text) else ""
                except Exception:
                    pred_text = str(pred_text)
            pred = str(pred_text)
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
                cache[fut[f][0]] = f.result()
                done += 1
                if done % 40 == 0:
                    json.dump(cache, open(CACHE_FILE, "w"), ensure_ascii=False, indent=1)
                    print(f"  ... {done}/{len(todo)} ({time.time() - t0:.0f}s)")
        json.dump(cache, open(CACHE_FILE, "w"), ensure_ascii=False, indent=1)
        print(f" judged in {time.time() - t0:.0f}s\n")

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
            e = cache.get(ck, {})
            print(f"[{m}] s{i} {e.get('verdict','?')}: gold={gold!r} pred={pred[:60]!r}")
            print(f"     reason: {e.get('reasoning','')}")


if __name__ == "__main__":
    main()
