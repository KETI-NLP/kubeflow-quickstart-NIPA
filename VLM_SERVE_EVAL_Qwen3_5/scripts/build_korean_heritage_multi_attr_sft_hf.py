"""
JSONL -> HF Arrow Dataset (DatasetDict with train/test).

build_korean_heritage_multi_attr_sft.py 가 만든 JSONL ([train 블록][test 블록] 순)을
HF DatasetDict 로 변환. 이미지는 Image(decode=True) 로 임베드 (사용자 선택: 옵션 A).

학습 파이프라인(VLM_SFT_Custom_Data_Qwen3_5)이 기대하는 schema:
  messages: list[{role, content}]
  images: Sequence(Image(decode=True))
"""

import argparse
import io
import json
import os
import time

from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image

DEFAULT_JSONL = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/tmp_eval_datasets/korean_heritage_multi_attr_sft.jsonl"
DEFAULT_OUT_DIR = "/workspace/2026_llm_data_generation/llm_training_ready/data_korean_heritage_multi_attr_sft_hf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default=DEFAULT_JSONL)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="JSONL 의 뒤쪽 N비율을 test 로 (default 0.1)")
    args = parser.parse_args()

    print(f"=== JSONL 로드: {args.jsonl} ===")
    t0 = time.time()
    records = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  {len(records)} records loaded in {time.time()-t0:.1f}s")

    # 1. unique 이미지 bytes 캐싱 (file IO 580k → 5.3k)
    unique_paths = list({r["image"] for r in records})
    print(f"\n=== unique 이미지 {len(unique_paths)}개 사전 로드 ===")
    t0 = time.time()
    img_bytes_cache = {}
    for i, p in enumerate(unique_paths):
        with open(p, "rb") as f:
            img_bytes_cache[p] = f.read()
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(unique_paths)}")
    total_img_size = sum(len(b) for b in img_bytes_cache.values()) / (1024**2)
    print(f"  이미지 캐시 완료: {time.time()-t0:.1f}s, 총 {total_img_size:.1f} MB in RAM")

    # 2. train/test split — JSONL 이 [train][test] 순서로 정렬돼 있음
    total = len(records)
    test_size = int(round(total * args.test_ratio))
    train_size = total - test_size
    train_records = records[:train_size]
    test_records = records[train_size:]
    print(f"\n=== split (JSONL 순서 기반): train {len(train_records)} / test {len(test_records)} ===")

    # 3. HF Dataset 스트리밍 빌드 (from_generator — 메모리에 한 번에 안 펼침)
    features = Features({
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "images": Sequence(Image(decode=True)),
    })

    def make_gen(recs):
        # img_bytes_cache 는 클로저로 캡처 (단일 프로세스)
        def _gen():
            for r in recs:
                yield {
                    "messages": r["messages"],
                    "images": [{"bytes": img_bytes_cache[r["image"]], "path": None}],
                }
        return _gen

    print(f"\n=== Dataset 빌드 (train, from_generator 스트리밍) ===")
    t0 = time.time()
    train_ds = Dataset.from_generator(make_gen(train_records), features=features)
    print(f"  train Dataset: {len(train_ds)} rows in {time.time()-t0:.1f}s")

    print(f"\n=== Dataset 빌드 (test, from_generator 스트리밍) ===")
    t0 = time.time()
    test_ds = Dataset.from_generator(make_gen(test_records), features=features)
    print(f"  test Dataset: {len(test_ds)} rows in {time.time()-t0:.1f}s")

    ds = DatasetDict({"train": train_ds, "test": test_ds})

    print(f"\n=== 디스크 저장 -> {args.out_dir} ===")
    t0 = time.time()
    if os.path.exists(args.out_dir):
        import shutil
        shutil.rmtree(args.out_dir)
    ds.save_to_disk(args.out_dir)
    print(f"  완료: {time.time()-t0:.1f}s")

    # 4. 검증
    print(f"\n=== 검증 ===")
    print(f"  features: {ds['train'].features}")
    print(f"  train sample[0] messages user: {ds['train'][0]['messages'][0]['content'][:80]!r}")
    print(f"  train sample[0] image type: {type(ds['train'][0]['images'][0]).__name__} {getattr(ds['train'][0]['images'][0],'size',None)}")

    # 디스크 크기
    import subprocess
    sz = subprocess.run(["du", "-sh", args.out_dir], capture_output=True, text=True).stdout.strip()
    print(f"  디스크 크기: {sz}")


if __name__ == "__main__":
    main()
