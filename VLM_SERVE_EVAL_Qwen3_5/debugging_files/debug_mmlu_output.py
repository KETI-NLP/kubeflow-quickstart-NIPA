"""
MMLU 커스텀 테스크 디버깅 스크립트
"""

import os
import sys
import yaml
from pathlib import Path

from datasets import load_dataset
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig, LiteLLMClient
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.default_prompts import LETTER_INDICES
from litellm import completion

sys.path.insert(0, str(Path(__file__).parent.parent / "custom_tasks"))
from custom_mmlu_task import create_custom_mmlu_task

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
            base_url=config['base_url'],
            api_key=config['api_key'],
            max_tokens=max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

def debug_mmlu_samples(subject: str = "abstract_algebra", num_samples: int = 5, yaml_path: str = None):
    if yaml_path is None:
        script_dir = Path(__file__).parent
        yaml_path = str(script_dir / "litellm_solar.yaml")
    
    print("=" * 80)
    print(f"MMLU 커스텀 테스크 디버깅: {subject}")
    print("=" * 80)
    print()
    
    print(" [1] LiteLLM 설정 로드 중...")
    litellm_config = load_litellm_config(str(yaml_path))
    print(f"     모델: {litellm_config['model_name']}")
    print(f"     Base URL: {litellm_config['base_url']}")
    print()
    
    print(" [2] 커스텀 테스크 생성 중...")
    task_config = create_custom_mmlu_task(subject)
    task = LightevalTask(config=task_config)
    print(f"     테스크 이름: {task_config.name}")
    print(f"     Metric: {task_config.metrics[0].metric_name}")
    print()
    
    print(" [3] 데이터셋 로드 중...")
    dataset = load_dataset("lighteval/mmlu", subject, split="test")
    print(f"     총 샘플 수: {len(dataset)}")
    print()
    
    print(" [4] 샘플 처리 중...")
    print("=" * 80)
    
    for i in range(min(num_samples, len(dataset))):
        record = dataset[i]
        
        doc = task_config.prompt_function(record, task_name=task_config.name)
        
        print(f"\n[샘플 {i+1}/{num_samples}]")
        print("-" * 80)
        print(f" 질문:")
        print(doc.query)
        print()
        print(f" 선택지:")
        for idx, choice in enumerate(doc.choices):
            marker = "✓" if idx == doc.gold_index else " "
            print(f"  {marker} {choice}")
        print()
        
        gold_indices = [doc.gold_index] if isinstance(doc.gold_index, int) else doc.gold_index
        gold_letters = [LETTER_INDICES[ix] for ix in gold_indices if 0 <= ix < len(LETTER_INDICES)]
        golds_from_choices = [g.strip() for g in doc.get_golds()]
        all_golds = list(set(gold_letters + golds_from_choices))
        print(f" 정답: {all_golds}")
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
                    score = output if isinstance(output, (int, float)) else output.get("score", 0.0) if isinstance(output, dict) else 0.0
                    print(f"   {metric.metric_name}: {score}")
                    print(f"   {'✅ 정답' if score > 0 else '❌ 오답'}")
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
    
    parser = argparse.ArgumentParser(description="MMLU 커스텀 테스크 디버깅")
    parser.add_argument(
        "--subject",
        type=str,
        default="abstract_algebra",
        help="MMLU 서브젝트 이름 (기본값: abstract_algebra)"
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
        help="LiteLLM YAML 설정 파일 경로 (기본값: litellm_solar.yaml)"
    )
    
    args = parser.parse_args()
    
    debug_mmlu_samples(subject=args.subject, num_samples=args.num_samples, yaml_path=args.yaml)
