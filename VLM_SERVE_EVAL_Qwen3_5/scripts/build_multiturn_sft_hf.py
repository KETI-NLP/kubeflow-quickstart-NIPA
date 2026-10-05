"""
멀티턴(멀티이미지) JSONL -> HF DatasetDict (train/test).

build_heritage_multiturn_sft.py 가 만든 JSONL 을 학습 파이프라인 schema 로 변환한다.
기존 build_korean_heritage_multi_attr_sft_hf.py 와 달리 레코드당 이미지가 여러 장
("images": [경로...])이며, messages content 의 <image> 마커 순서와 1:1 대응한다.

- split 은 레코드의 "split" 필드("train"/"test")로 결정 (paraphrase 수와 무관하게
  heritage-aware 분리가 정확히 보존됨).
- unique 이미지 bytes 를 사전 캐싱해 file IO 를 줄이고, from_generator 스트리밍으로
  메모리에 한 번에 펼치지 않는다.

학습 파이프라인(VLM_SFT_Custom_Data_Qwen3_5)이 기대하는 schema:
  messages: list[{role, content}]
  images:   Sequence(Image(decode=True))   # 레코드당 N장
"""

import argparse
import io
import json
import os
import time

from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image


def load_resized_jpeg_bytes(path: str, max_side: int, quality: int = 90) -> bytes:
    """이미지를 긴 변 max_side 로 다운스케일 후 JPEG bytes 로 반환.
    학습 collator 가 어차피 512px 로 리사이즈하므로 사전 리사이즈해도 학습 손실 0,
    저장 용량만 크게 절약된다(원본은 수 MB 인 고해상도도 있음)."""
    img = PILImage.open(path).convert("RGB")
    w, h = img.size
    if max_side and max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         PILImage.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-side", type=int, default=512,
                        help="이미지 긴 변 최대 픽셀(저장 용량 절감). 0=리사이즈 안 함.")
    args = parser.parse_args()

    print(f"=== JSONL 로드: {args.jsonl} ===")
    t0 = time.time()
    train_records, test_records = [], []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            (test_records if r.get("split") == "test" else train_records).append(r)
    print(f"  train {len(train_records)} / test {len(test_records)} loaded in {time.time()-t0:.1f}s")

    # 1. unique 이미지 bytes 캐싱
    unique_paths = set()
    for r in train_records + test_records:
        for p in r["images"]:
            unique_paths.add(p)
    unique_paths = list(unique_paths)
    print(f"\n=== unique 이미지 {len(unique_paths)}개 사전 로드 (긴 변 ≤{args.max_side or '원본'}px) ===")
    t0 = time.time()
    img_bytes_cache = {}
    for i, p in enumerate(unique_paths):
        img_bytes_cache[p] = load_resized_jpeg_bytes(p, args.max_side)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(unique_paths)}")
    total_mb = sum(len(b) for b in img_bytes_cache.values()) / (1024**2)
    print(f"  완료: {time.time()-t0:.1f}s, 총 {total_mb:.1f} MB in RAM")

    # image_urls: 저장된 각 이미지의 원본 URL(순서 일치). 배포 시 이미지 bytes 를 빼고
    # URL 만 남겨 재다운로드할 수 있게 한다. 학습에는 안 쓰이지만 remove_unused_columns=False
    # 라서 collator 가 무시할 뿐 에러는 없다(검증 완료).
    features = Features({
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "images": Sequence(Image(decode=True)),
        "image_urls": Sequence(Value("string")),
    })

    def make_gen(recs):
        def _gen():
            for r in recs:
                yield {
                    "messages": r["messages"],
                    "images": [{"bytes": img_bytes_cache[p], "path": None} for p in r["images"]],
                    "image_urls": r.get("image_urls", []),
                }
        return _gen

    print(f"\n=== Dataset 빌드 (from_generator 스트리밍) ===")
    t0 = time.time()
    train_ds = Dataset.from_generator(make_gen(train_records), features=features)
    if test_records:
        test_ds = Dataset.from_generator(make_gen(test_records), features=features)
        ds = DatasetDict({"train": train_ds, "test": test_ds})
        print(f"  train {len(train_ds)} / test {len(test_ds)} rows in {time.time()-t0:.1f}s")
    else:
        ds = DatasetDict({"train": train_ds})
        print(f"  train {len(train_ds)} rows (no test split) in {time.time()-t0:.1f}s")

    print(f"\n=== 디스크 저장 -> {args.out_dir} ===")
    t0 = time.time()
    if os.path.exists(args.out_dir):
        import shutil
        shutil.rmtree(args.out_dir)
    ds.save_to_disk(args.out_dir)
    print(f"  완료: {time.time()-t0:.1f}s")

    print(f"\n=== 검증 ===")
    print(f"  features: {ds['train'].features}")
    s = ds["train"][0]
    print(f"  sample[0] images: {len(s['images'])}장, sizes={[getattr(im,'size',None) for im in s['images']]}")
    for m in s["messages"]:
        print(f"    [{m['role']}] {m['content'][:80]!r}")
    n_img_markers = sum(m["content"].count("<image>") for m in s["messages"])
    print(f"  <image> 마커 수={n_img_markers} vs images={len(s['images'])} "
          f"-> {'OK' if n_img_markers == len(s['images']) else 'MISMATCH!!'}")

    import subprocess
    sz = subprocess.run(["du", "-sh", args.out_dir], capture_output=True, text=True).stdout.strip()
    print(f"  디스크 크기: {sz}")


if __name__ == "__main__":
    main()
