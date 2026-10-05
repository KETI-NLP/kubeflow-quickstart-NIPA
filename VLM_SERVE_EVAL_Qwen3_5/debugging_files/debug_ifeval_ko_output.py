"""
IFEval-Ko 커스텀 테스크 디버깅 스크립트
"""

import os
import sys
import yaml
import glob
from pathlib import Path

from datasets import load_dataset
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig, LiteLLMClient
from lighteval.models.model_output import ModelResponse
from litellm import completion

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

sys.path.insert(0, str(Path(__file__).parent.parent / "custom_tasks"))
from custom_ifeval_ko_task import create_custom_ifeval_ko_task

def load_litellm_config(yaml_path: str):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['model_parameters']

def call_model(prompt: str, config: dict) -> str:
    try:
        max_tokens = config.get('max_model_length', 8192)
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

def debug_ifeval_ko_samples(num_samples: int = 5, output_dir: str = "./test_ifeval_ko_solar", use_cache: bool = True):
    
    script_dir = Path(__file__).parent
    yaml_path = script_dir / "litellm_solar.yaml"
    
    print("=" * 80)
    print("IFEval-Ko 커스텀 테스크 디버깅")
    print("=" * 80)
    print()
    
    print(" [1] LiteLLM 설정 로드 중...")
    litellm_config = load_litellm_config(str(yaml_path))
    print(f"     모델: {litellm_config['model_name']}")
    print(f"     Base URL: {litellm_config['base_url']}")
    print()
    
    print(" [2] 커스텀 테스크 생성 중...")
    task_config = create_custom_ifeval_ko_task()
    task = LightevalTask(config=task_config)
    task_name = task_config.name
    print(f"     테스크 이름: {task_name}")
    print(f"     Metric: {task_config.metrics[0].metric_name}")
    print()
    
    print(" [3] 데이터셋 로드 중...")
    dataset = load_dataset("allganize/IFEval-Ko", split="train")
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
        record_key = None
        instruction_ids = None
        kwargs_list = None
        
        if hasattr(doc, 'specific') and doc.specific:
            record_key = doc.specific.get('key')
            instruction_ids = doc.specific.get('instruction_id_list')
            kwargs_list = doc.specific.get('kwargs')
        
        if not record_key and hasattr(doc, 'key'):
            record_key = doc.key
        if not instruction_ids and hasattr(doc, 'instruction_id_list'):
            instruction_ids = doc.instruction_id_list
        if not kwargs_list and hasattr(doc, 'kwargs'):
            kwargs_list = doc.kwargs
        
        print(f"\n[샘플 {sample_idx+1}/{num_samples}]")
        print("-" * 80)
        if record_key:
            print(f" Key: {record_key}")
        if instruction_ids:
            print(f" Instruction IDs: {instruction_ids}")
        print()
        print(f" 프롬프트:")
        print(doc.query)
        print()
        
        if kwargs_list:
            print(f" 평가 파라미터 (kwargs):")
            for j, kwarg in enumerate(kwargs_list):
                if isinstance(kwarg, dict):
                    non_none = {k: v for k, v in kwarg.items() if v is not None}
                    if non_none:
                        print(f"   [{j}]: {non_none}")
            print()
        
        if cached_results is not None and sample_idx < len(cached_results):
            print(" lighteval 캐시에서 모델 출력 로드 중...")
            model_output = cached_results.iloc[sample_idx]['final_text']
            cached_metrics = {}
            for metric_name in ['prompt_level_strict_acc', 'inst_level_strict_acc', 
                              'prompt_level_loose_acc', 'inst_level_loose_acc']:
                if metric_name in cached_results.columns:
                    cached_metrics[metric_name] = cached_results.iloc[sample_idx][metric_name]
            if cached_metrics:
                print(f"     캐시된 메트릭: {cached_metrics}")
        else:
            print(" 모델 호출 중...")
            model_output = call_model(doc.query, litellm_config)
        
        print(f" 모델 출력:")
        print(f"   '{model_output}' (전체 길이: {len(model_output)} 문자)")
        print()
        
        model_response = ModelResponse(text=[model_output])
        
        print(f" Metric 결과:")
        for metric in task_config.metrics:
            if metric.category.value == "GENERATIVE":
                try:
                    output = metric.sample_level_fn.compute(doc, model_response)
                    if isinstance(output, dict):
                        for key, value in output.items():
                            if isinstance(value, list):
                                true_count = sum(value)
                                total_count = len(value)
                                print(f"   {key}: {true_count}/{total_count} ({[int(v) for v in value]})")
                            else:
                                print(f"   {key}: {value}")
                    else:
                        print(f"   {metric.metric_name}: {output}")
                except Exception as e:
                    import traceback
                    print(f"   {metric.metric_name}: 계산 실패 - {e}")
                    traceback.print_exc()
        print()
        
        print("=" * 80)
    
    print("\n 디버깅 완료!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IFEval-Ko 테스크 디버깅")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="확인할 샘플 수 (기본값: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_ifeval_ko_solar",
        help="lighteval의 출력 디렉토리 (캐시된 결과를 읽기 위해, 기본값: ./test_ifeval_ko_solar)"
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
    
    debug_ifeval_ko_samples(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        use_cache=not args.no_cache,
        yaml_path=args.yaml
    )

