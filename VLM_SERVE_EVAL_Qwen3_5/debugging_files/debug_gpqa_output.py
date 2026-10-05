"""
GPQA 커스텀 테스크 디버깅 스크립트
"""

import os
import sys
import yaml
import random
from pathlib import Path
from string import ascii_uppercase


from datasets import load_dataset
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig, LiteLLMClient
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.default_prompts import LETTER_INDICES
from litellm import completion


sys.path.insert(0, str(Path(__file__).parent.parent / "custom_tasks"))
from custom_gpqa_task import create_custom_gpqa_task, record_to_sample
from lighteval.metrics.metrics_sample import SampleLevelComputation

def load_litellm_config(yaml_path: str):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['model_parameters']

def call_model(prompt: str, config: dict) -> str:
    try:
        max_tokens = config.get('max_model_length', 15000)
        response = completion(
            model=config['model_name'],
            messages=[{"role": "user", "content": prompt}],
            api_base=config['base_url'],
            api_key=config['api_key'],
            max_tokens=max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

def debug_gpqa_samples(subset: str = "gpqa_main", num_samples: int = 5, yaml_path: str = None):
    """GPQA 샘플 디버깅"""
    
    if yaml_path is None:
        script_dir = Path(__file__).parent
        yaml_path = str(script_dir / "litellm_qwen3.yaml")
    
    print("=" * 80)
    print(f"GPQA 커스텀 테스크 디버깅: {subset}")
    print("=" * 80)
    print()
    
    print(" [1] LiteLLM 설정 로드 중...")
    litellm_config = load_litellm_config(str(yaml_path))
    print(f"     모델: {litellm_config['model_name']}")
    print(f"     Base URL: {litellm_config['base_url']}")
    print()
    
    print(" [2] 커스텀 테스크 생성 중...")
    task_config = create_custom_gpqa_task(subset)
    task = LightevalTask(config=task_config)
    print(f"     테스크 이름: {task_config.name}")
    print(f"     Metric: {task_config.metrics[0].metric_name}")
    print()
    
    print(" [3] 데이터셋 로드 중...")
    dataset = load_dataset("Idavidrein/gpqa", subset, split="train")
    print(f"     총 샘플 수: {len(dataset)}")
    print()
    
    print(" [4] 샘플 처리 중...")
    print("=" * 80)
    
    for i in range(min(num_samples, len(dataset))):
        record = dataset[i]
        
        sample = record_to_sample(record)
        
        doc = task_config.prompt_function(sample, task_name=task_config.name)
        
        print(f"\n[샘플 {i+1}/{num_samples}]")
        print("-" * 80)
        
        print(f" 원본 정답: {sample['Correct Answer']}")
        print()
        
        print(f" 질문:")
        print(doc.query)
        print()
        
        print(f" 선택지 (섞인 순서):")
        gold_indices = [doc.gold_index] if isinstance(doc.gold_index, int) else doc.gold_index
        gold_letters = [LETTER_INDICES[ix] for ix in gold_indices if 0 <= ix < len(LETTER_INDICES)]
        
        query_lines = doc.query.split("\n")
        for line in query_lines:
            if line.strip().startswith(("A)", "B)", "C)", "D)")):
                letter = line.strip()[0]
                idx = ord(letter) - ord("A")
                marker = "✓" if idx in gold_indices else " "
                print(f"  {marker} {line.strip()}")
        print()
        
        golds_from_choices = [g.strip() for g in doc.get_golds()]
        all_golds = list(set(gold_letters + golds_from_choices))
        print(f" 정답 (섞인 후 위치): {all_golds} (인덱스: {doc.gold_index})")
        print()
        
        print(" 모델 호출 중...")
        model_output = call_model(doc.query, litellm_config)
        print(f" 모델 출력:")
        print(f"   '{model_output}'")
        print()
        
        model_response = ModelResponse(text=[model_output])
        
        print(f" Metric 결과:")
        for metric in task_config.metrics:
            if metric.category.value == "GENERATIVE":
                try:
                    output = metric.sample_level_fn.compute(doc, model_response)
                    if isinstance(output, (int, float)):
                        score = float(output)
                    elif isinstance(output, dict):
                        score = output.get("score", output.get(list(output.keys())[0] if output else "score", 0.0))
                    else:
                        score = 0.0
                    print(f"   {metric.metric_name}: {score}")
                    print(f"   {'✅ 정답 포함' if score > 0 else '❌ 정답 없음'}")
                except Exception as e:
                    print(f"   {metric.metric_name}: 계산 실패 - {e}")
        print()
        
        print(" 상세 분석:")
        for gold in all_golds:
            pred_upper = model_output.upper()
            gold_upper = gold.upper()
            contains = gold_upper in pred_upper
            print(f"   정답 '{gold}' {'포함됨' if contains else '포함 안됨'}")
            if contains:
                idx = pred_upper.find(gold_upper)
                context = model_output[max(0, idx-20):idx+len(gold)+20]
                print(f"      위치: ...{context}...")
        print()
        
        print("=" * 80)
    
    print("\n 디버깅 완료!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPQA 커스텀 테스크 디버깅")
    parser.add_argument(
        "--subset",
        type=str,
        default="gpqa_main",
        help="GPQA 서브셋 이름 (기본값: gpqa_main, 옵션: gpqa_main, gpqa_extended, gpqa_diamond)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="확인할 샘플 수 (기본값: 5)"
    )
    
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="LiteLLM YAML 설정 파일 경로 (기본값: litellm_qwen3.yaml)"
    )
    
    args = parser.parse_args()
    
    debug_gpqa_samples(subset=args.subset, num_samples=args.num_samples, yaml_path=args.yaml)

