"""
MMBench 커스텀 테스크 디버깅 스크립트 (Prompt 및 Image 파싱 확인용)
"""

import sys
from pathlib import Path
from datasets import load_dataset
from lighteval.tasks.lighteval_task import LightevalTask

sys.path.insert(0, str(Path(__file__).parent.parent / "custom_tasks"))
from custom_mmbench_task import create_custom_mmbench_task

def debug_mmbench_samples(subset: str = "en", num_samples: int = 5):
    print("=" * 80)
    print(f"MMBench 커스텀 테스크 디버깅: {subset}")
    print("=" * 80)
    print()
    
    print(" [1] 커스텀 테스크 생성 중...")
    task_config = create_custom_mmbench_task(subset)
    task = LightevalTask(config=task_config)
    print(f"     테스크 이름: {task_config.name}")
    print()
    
    print(" [2] 데이터셋 로드 및 샘플 선택 중...")
    selected_docs = task.get_docs(max_samples=num_samples)
    print(f"     선택된 샘플 수: {len(selected_docs)}")
    print()
    
    print(" [3] 샘플 내용 확인 중...")
    print("=" * 80)
    
    for sample_idx, doc in enumerate(selected_docs):
        print(f"\n[샘플 {sample_idx+1}/{num_samples}]")
        print("-" * 80)
        print(f" 질문 텍스트:")
        print(doc.query)
        print()
        
        print(f" 이미지 데이터 여부:")
        if getattr(doc, "images", None):
            print(f"   => {len(doc.images)} 개의 이미지가 성공적으로 로드되었습니다.")
            for i, img in enumerate(doc.images):
                print(f"      - Image {i+1}: {type(img)}")
        else:
            print("   => 이미지가 없습니다.")
        print()
        
        print(f" 선택지 (있을 경우):")
        for idx, choice in enumerate(doc.choices):
            marker = "✓" if (isinstance(doc.gold_index, int) and idx == doc.gold_index) or (isinstance(doc.gold_index, list) and idx in doc.gold_index) else " "
            print(f"  {marker} {choice}")
        print()
        
        print("=" * 80)
    
    print("\n 디버깅 완료!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="en", help="MMBench subset")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to check")
    args = parser.parse_args()
    
    debug_mmbench_samples(subset=args.subset, num_samples=args.num_samples)
