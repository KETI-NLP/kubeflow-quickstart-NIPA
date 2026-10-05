"""
KMMLU 커스텀 테스크 디버깅 스크립트
"""

import os
import sys
import yaml
import random
import json
import glob
from pathlib import Path

from datasets import load_dataset
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig, LiteLLMClient
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.default_prompts import LETTER_INDICES
from litellm import completion
import litellm

litellm.cache = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

sys.path.insert(0, str(Path(__file__).parent.parent / "custom_tasks"))
from custom_kmmlu_task import create_custom_kmmlu_task

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
            caching=False,  
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

def load_lighteval_cache(output_dir: str, task_name: str):
    if not HAS_PANDAS:
        return None
    
    patterns = [
        f"{output_dir}/details/**/*{task_name}*.parquet",  
        f"{output_dir}/details/**/*{task_name.replace(':', '_')}*.parquet",  
        f"{output_dir}/details/**/*details_*{task_name.replace(':', '_')}*.parquet",  
    ]
    
    parquet_files = []
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        if found:
            parquet_files.extend(found)
            break  
    
    if not parquet_files:
        task_base = task_name.split(':')[0] if ':' in task_name else task_name
        pattern = f"{output_dir}/details/**/*{task_base}*.parquet"
        parquet_files = glob.glob(pattern, recursive=True)
    
    if not parquet_files:
        return None
    
    latest_file = sorted(parquet_files, key=lambda x: os.path.getmtime(x))[-1]
    try:
        df = pd.read_parquet(latest_file)
        if 'final_text' in df.columns:
            print(f"     캐시 파일 로드: {latest_file}")
            return df
    except Exception as e:
        print(f"     캐시 파일 읽기 실패: {e}")
        return None
    
    return None

def debug_kmmlu_samples(subject: str = "Accounting", num_samples: int = 5, output_dir: str = "./test_kmmlu_solartm", use_cache: bool = True, yaml_path: str = None):
    if yaml_path is None:
        script_dir = Path(__file__).parent
        yaml_path = str(script_dir / "litellm_solar.yaml")
    
    print("=" * 80)
    print(f"KMMLU 커스텀 테스크 디버깅: {subject}")
    print("=" * 80)
    print()
    
    print(" [1] LiteLLM 설정 로드 중...")
    litellm_config = load_litellm_config(str(yaml_path))
    print(f"     모델: {litellm_config['model_name']}")
    print(f"     Base URL: {litellm_config['base_url']}")
    print()
    
    print(" [2] 커스텀 테스크 생성 중...")
    task_config = create_custom_kmmlu_task(subject)
    task = LightevalTask(config=task_config)
    task_name = task_config.name
    print(f"     테스크 이름: {task_name}")
    print(f"     Metric: {task_config.metrics[0].metric_name}")
    print()
    
    print(" [3] 데이터셋 로드 중...")
    dataset = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")
    print(f"     총 샘플 수: {len(dataset)}")
    print()
    
    print(" [4] 샘플 선택 중 (lighteval과 동일한 방식: task.get_docs 사용)...")
    selected_docs = task.get_docs(max_samples=num_samples)  
    
    print(f"     선택된 샘플 수: {len(selected_docs)}")
    print()
    
    cached_results = None
    if use_cache and HAS_PANDAS:
        print(" [5] lighteval 캐시 로드 중...")
        cached_results = load_lighteval_cache(output_dir, task_name)
        if cached_results is not None:
            print(f"     캐시된 결과 로드 성공: {len(cached_results)} 샘플")
        else:
            print(f"     캐시된 결과를 찾을 수 없습니다. 모델을 직접 호출합니다.")
        print()
    
    print(" [6] 샘플 처리 중...")
    print("=" * 80)
    
    for sample_idx, doc in enumerate(selected_docs):
        print(f"\n[샘플 {sample_idx+1}/{num_samples}]")
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

        model_output = None
        if cached_results is not None:
            cache_idx = None
            if 'query' in cached_results.columns:
                matching = cached_results['query'] == doc.query
                if matching.any():
                    cache_idx = matching.idxmax()  
            elif sample_idx < len(cached_results):
                cache_idx = sample_idx
            
            if cache_idx is not None:
                print(" lighteval 캐시에서 모델 출력 로드 중...")
                model_output = cached_results.iloc[cache_idx]['final_text']
                cached_metrics = {}
                for metric_name in ['extractive_match']:
                    if metric_name in cached_results.columns:
                        cached_metrics[metric_name] = cached_results.iloc[cache_idx][metric_name]
                if cached_metrics:
                    print(f"     캐시된 메트릭: {cached_metrics}")
        
        if model_output is None:
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
                    print(f"   {'✅ 정답' if score > 0 else '❌ 오답'}")
                    if hasattr(doc, 'specific') and doc.specific:
                        extracted_preds = doc.specific.get('extracted_predictions', [])
                        extracted_golds = doc.specific.get('extracted_golds', [])
                        if extracted_preds or extracted_golds:
                            print(f"   추출된 예측: {extracted_preds}")
                            print(f"   추출된 정답: {extracted_golds}")
                except Exception as e:
                    print(f"   {metric.metric_name}: 계산 실패 - {e}")
                    import traceback
                    traceback.print_exc()
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
    
    parser = argparse.ArgumentParser(description="KMMLU 커스텀 테스크 디버깅")
    parser.add_argument(
        "--subject",
        type=str,
        default="Accounting",
        help="KMMLU 서브젝트 이름 (기본값: Accounting)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="확인할 샘플 수 (기본값: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_kmmlu_solartm",
        help="lighteval의 출력 디렉토리 (캐시된 결과를 읽기 위해, 기본값: ./test_kmmlu_solartm)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="lighteval 캐시를 사용하지 않고 모델을 직접 호출"
    )
    
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="LiteLLM YAML 설정 파일 경로 (기본값: litellm_solar.yaml)"
    )
    
    args = parser.parse_args()
    
    debug_kmmlu_samples(
        subject=args.subject, 
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        use_cache=not args.no_cache,
        yaml_path=args.yaml
    )

