"""
GSM8K 테스크 디버깅 스크립트
"""

import os
import sys
import yaml
from pathlib import Path
import time

from datasets import load_dataset
from lighteval.tasks.registry import Registry
from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig, LiteLLMClient
from lighteval.models.model_output import ModelResponse
from litellm import completion

sys.path.insert(0, str(Path(__file__).parent.parent / "custom_tasks"))
from custom_gsm8k_task import custom_gsm8k_prompt, create_custom_gsm8k_task

def load_litellm_config(yaml_path: str):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['model_parameters']

def call_model(prompt: str, config: dict) -> str:
    try:
        max_tokens = config.get('max_model_length', 4000)
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

def debug_gsm8k_samples(num_samples: int = 5, yaml_path: str = None):
    if yaml_path is None:
        script_dir = Path(__file__).parent
        yaml_path = str(script_dir / "litellm_solar.yaml")
    
    print("=" * 80)
    print("GSM8K 테스크 디버깅")
    print("=" * 80)
    print()
    
    print(" [1] LiteLLM 설정 로드 중...")
    litellm_config = load_litellm_config(str(yaml_path))
    print(f"     모델: {litellm_config['model_name']}")
    print(f"     Base URL: {litellm_config['base_url']}")
    print()
    
    print(" [2] 커스텀 GSM8K 테스크 로드 중...")
    task = create_custom_gsm8k_task()
    
    print(f"     테스크 이름: {task.name}")
    print(f"     Metrics: {[m.metric_name for m in task.metrics]}")
    print()
    
    print(" [3] 데이터셋 로드 중...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"     총 샘플 수: {len(dataset)}")
    print()
    
    print(" [4] 샘플 처리 중...")
    print("=" * 80)
    
    for i in range(min(num_samples, len(dataset))):
        record = dataset[i]
        
        doc = custom_gsm8k_prompt(record, task_name=task.name)
        
        print(f"\n[샘플 {i+1}/{num_samples}]")
        print("-" * 80)
        print(f" 질문:")
        print(doc.query)
        print()
        
        golds = doc.get_golds()
        print(f" 정답: {golds}")
        print()
        
        print(" 모델 호출 중...")
        model_output = call_model(doc.query, litellm_config)
        print(f" 모델 출력:")
        print(f"   '{model_output}'")
        print()
        
        model_response = ModelResponse(text=[model_output])
        
        print(f" Metric 결과:")
        for metric in task.metrics:
            if metric.category.value == "GENERATIVE":
                try:
                    output = metric.sample_level_fn.compute(doc, model_response)
                    score = output if isinstance(output, (int, float)) else output.get("score", 0.0) if isinstance(output, dict) else 0.0
                    print(f"   {metric.metric_name}: {score}")
                    print(f"   {'✅ 정답' if score > 0 else '❌ 오답'}")
                except Exception as e:
                    print(f"   {metric.metric_name}: 계산 실패 - {e}")
        print()
        
        print("=" * 80)
    
    print("\n 디버깅 완료!")

if __name__ == "__main__":
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser(description="GSM8K 테스크 디버깅")
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
    
    debug_gsm8k_samples(num_samples=args.num_samples, yaml_path=args.yaml)

    end_time = time.time()
    print(f"디버깅 소요 시간: {end_time - start_time} 초")