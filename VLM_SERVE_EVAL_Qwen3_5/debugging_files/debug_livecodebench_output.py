"""
LiveCodeBench Qwen 커스텀 테스크 디버깅 스크립트
"""

import os
import sys
import yaml
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, TextIO

from datasets import load_dataset
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.models.model_output import ModelResponse
from litellm import completion

sys.path.insert(0, str(Path(__file__).parent.parent / "custom_tasks"))
from custom_livecodebench_filtered_task_qwen import (
    create_custom_livecodebench_qwen_task,
    CodegenMetricFiltered,
    filter_by_date,
    parse_date,
    START_DATE,
    END_DATE,
    extract_code_qwen,
    translate_private_test_cases,
)


class TeeOutput:
    def __init__(self, *files: TextIO):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def load_litellm_config(yaml_path: str):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    params = config['model_parameters']
    gen_params = params.get('generation_parameters', {})
    
    return {
        'model_name': params['model_name'],
        'base_url': params.get('base_url'),
        'api_key': params.get('api_key'),
        'generation_parameters': gen_params,
    }


def call_model(prompt: str, config: dict) -> str:
    try:
        gen_params = config.get('generation_parameters', {})
        max_tokens = config.get('max_model_length') or gen_params.get('max_tokens', 8192)
        response = completion(
            model=config['model_name'],
            messages=[{"role": "user", "content": prompt}],
            api_base=config.get('base_url'),
            api_key=config.get('api_key'),
            max_tokens=max_tokens,
            temperature=gen_params.get('temperature', 0.0),
            **{k: v for k, v in gen_params.items() if k not in ['max_tokens', 'temperature']}
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"


def debug_livecodebench_qwen_samples(
    num_samples: int = 3,
    output_file: Optional[str] = None,
    yaml_path: str = None,
    use_date_filter: bool = False,
    subset: str = "v5",
):
    """
    LiveCodeBench Qwen 샘플 디버깅
    
    Args:
        num_samples: 디버깅할 샘플 수
        output_file: 출력 파일 경로
        yaml_path: LiteLLM 설정 파일 경로
        use_date_filter: True면 날짜 필터링 사용, False면 v5 subset 전체 사용
        subset: 사용할 subset ("v5" 등)
    """
    
    output_handle = None
    original_stdout = sys.stdout
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = open(output_path, 'w', encoding='utf-8')
        sys.stdout = TeeOutput(original_stdout, output_handle)
        print(f"출력을 파일에 저장합니다: {output_path}")
        print("=" * 80)
        print()
    
    try:
        if yaml_path is None:
            script_dir = Path(__file__).parent
            yaml_path = str(script_dir / "litellm_qwen3.yaml")
        else:
            yaml_path = str(yaml_path)
        
        print("=" * 80)
        print(f"LiveCodeBench Qwen 커스텀 테스크 디버깅")
        if use_date_filter:
            print(f"날짜 필터: {START_DATE.date()} ~ {END_DATE.date()}")
            print(f"(전체 데이터셋에서 필터링)")
        else:
            print(f"Subset: {subset} (전체 사용)")
        print("=" * 80)
        print()
        
        print(" [1] LiteLLM 설정 로드 중...")
        litellm_config = load_litellm_config(str(yaml_path))
        print(f"     모델: {litellm_config['model_name']}")
        print(f"     Base URL: {litellm_config.get('base_url', 'N/A')}")
        print(f"     Generation params: {litellm_config.get('generation_parameters', {})}")
        print()
        
        print(" [2] 커스텀 테스크 생성 중...")
        task_config = create_custom_livecodebench_qwen_task(
            subset=subset,
            use_date_filter=use_date_filter
        )
        task = LightevalTask(config=task_config)
        print(f"     테스크 이름: {task_config.name}")
        print(f"     Metric: {task_config.metrics[0].metric_name}")
        print(f"     Subset: {task_config.hf_subset} ({'전체 데이터셋' if task_config.hf_subset is None else task_config.hf_subset})")
        print(f"     Date filter: {'사용' if use_date_filter else '사용 안 함'}")
        print()
        
        print(" [3] 데이터셋 로드 중...")
        # task_config의 hf_subset을 사용 (날짜 필터링 시 None = 전체 데이터셋)
        actual_subset = task_config.hf_subset
        if actual_subset is None:
            dataset = load_dataset("lighteval/code_generation_lite", split="test")
            print(f"     전체 데이터셋: {len(dataset)}개")
        else:
            dataset = load_dataset("lighteval/code_generation_lite", actual_subset, split="test")
            print(f"     Subset ({actual_subset}): {len(dataset)}개")
        
        if use_date_filter:
            filtered_samples = []
            for record in dataset:
                if filter_by_date(record, START_DATE, END_DATE):
                    filtered_samples.append(record)
            print(f"     날짜 필터링 후: {len(filtered_samples)}개")
        else:
            print(f"     필터링 없음: {len(dataset)}개")
        print()
        
        metric = CodegenMetricFiltered()
        
        print(" [4] 샘플 처리 중...")
        print("=" * 80)
        
        sample_count = 0
        for record in dataset:
            if sample_count >= num_samples:
                break
            
            if use_date_filter:
                if not filter_by_date(record, START_DATE, END_DATE):
                    continue
            
            sample_count += 1
            
            doc = task_config.prompt_function(record, task_name=task_config.name)
            
            if not doc or not doc.query or doc.specific is None:
                continue
            
            print(f"\n[샘플 {sample_count}/{num_samples}]")
            print("-" * 80)
            
            question_id = record.get("question_id", "N/A")
            contest_date = record.get("contest_date", "N/A")
            platform = record.get("platform", "N/A")
            difficulty = record.get("difficulty", "N/A")
            
            print(f" Question ID: {question_id}")
            print(f" Contest Date: {contest_date}")
            print(f" Platform: {platform}")
            print(f" Difficulty: {difficulty}")
            print()
            
            if use_date_filter:
                parsed_date = parse_date(contest_date)
                in_range = START_DATE <= parsed_date < END_DATE if parsed_date else False
                print(f" 날짜 필터링: {'✅ 통과' if in_range else '❌ 거절'}")
                if parsed_date:
                    print(f"   날짜: {parsed_date.date()}")
                    print(f"   범위 내: {START_DATE.date()} <= {parsed_date.date()} < {END_DATE.date()}")
                print()
            
            question_content = record.get("question_content", "")
            print(f" 문제 내용 (처음 200자):")
            print(f"   {question_content[:200]}...")
            print()
            
            starter_code = record.get("starter_code")
            if starter_code:
                print(f" Starter Code (처음 200자):")
                print(f"   {starter_code[:200]}...")
                print()
            
            print(f" 프롬프트:")
            print(f"   {doc.query}...")
            print()
            
            inputs = doc.specific.get("inputs", [])
            outputs = doc.specific.get("outputs", [])
            print(f" 테스트 케이스:")
            print(f"   총 {len(inputs)}개 테스트 케이스")
            if len(inputs) > 0:
                print(f"   첫 번째 테스트 입력:")
                print(f"     {inputs[0][:200] if len(inputs[0]) > 200 else inputs[0]}...")
                print(f"   첫 번째 테스트 출력:")
                print(f"     {outputs[0][:200] if outputs and len(outputs[0]) > 200 else (outputs[0] if outputs else 'N/A')}...")
            print()
            
            print(" 모델 호출 중...")
            model_output = call_model(doc.query, litellm_config)
            print(f" 모델 출력 (길이: {len(model_output)} 문자)")
            print(f"   처음 500자:")
            print(f"   {model_output[:500]}...")
            print()
            print(f"   마지막 500자:")
            print(f"   ...{model_output[-500:]}")
            print()
            
            extracted_code = extract_code_qwen(model_output)
            print(f" 추출된 코드 (길이: {len(extracted_code)} 문자):")
            if extracted_code:
                print(f"   {extracted_code[:500]}..." if len(extracted_code) > 500 else f"   {extracted_code}")
                print()
                print(f"   코드 블록 분석:")
                code_lines = extracted_code.split("\n")
                print(f"     총 줄 수: {len(code_lines)}")
                print(f"     import 문: {sum(1 for line in code_lines if line.strip().startswith('import ') or line.strip().startswith('from '))}개")
                print(f"     stdin/stdout 사용: {'input()' in extracted_code or 'sys.stdin' in extracted_code or 'readline' in extracted_code or 'print(' in extracted_code}")
            else:
                print(f"   ❌ 코드 추출 실패")
            print()
            
            # 실제 metric에서 사용하는 방식으로 코드 추출 확인
            print(f" Metric 내부 코드 추출 확인:")
            model_response_for_check = ModelResponse(text=[model_output])
            if hasattr(model_response_for_check, 'final_text'):
                final_texts = model_response_for_check.final_text
                print(f"   final_text 개수: {len(final_texts) if isinstance(final_texts, list) else 'N/A'}")
                for idx, ft in enumerate(final_texts[:3]):  # 처음 3개만 확인
                    extracted_from_final = extract_code_qwen(ft)
                    print(f"   final_text[{idx}]에서 추출된 코드 길이: {len(extracted_from_final)}")
                    if extracted_from_final:
                        print(f"     처음 100자: {extracted_from_final[:100]}...")
            print()
            
            # ModelResponse 생성 - 실제 lighteval과 동일하게
            model_response = ModelResponse(text=[model_output])
            
            # final_text 확인 (실제 metric에서 사용하는 값)
            print(f" ModelResponse.final_text 확인:")
            print(f"   타입: {type(model_response.final_text)}")
            print(f"   길이: {len(model_response.final_text) if isinstance(model_response.final_text, list) else 'N/A'}")
            if isinstance(model_response.final_text, list) and len(model_response.final_text) > 0:
                print(f"   첫 번째 요소 길이: {len(model_response.final_text[0])}")
                print(f"   첫 번째 요소 처음 100자: {model_response.final_text[0][:100]}...")
            print()
            
            try:
                score = metric.compute(model_response, doc)
                print(f" Metric 결과:")
                print(f"   Score: {score}")
                print(f"   {'✅ 통과' if score > 0 else '❌ 실패'}")
                print()
            except Exception as e:
                print(f" Metric 계산 실패: {e}")
                import traceback
                traceback.print_exc()
                print()
            
            if extracted_code and inputs and outputs:
                print(f" 테스트 케이스 정보:")
                print(f"   총 {len(inputs)}개 테스트 케이스")
                print()
            
            print("=" * 80)
        
        print(f"\n 디버깅 완료! (처리된 샘플: {sample_count}개)")
        if output_file:
            print(f" 출력 파일: {output_file}")
    
    finally:
        if output_handle:
            sys.stdout = original_stdout
            output_handle.close()
            print(f"\n출력이 파일에 저장되었습니다: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LiveCodeBench Qwen 커스텀 테스크 디버깅")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="확인할 샘플 수 (기본값: 3)"
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="LiteLLM YAML 설정 파일 경로 (기본값: litellm_qwen3.yaml)"
    )
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        default=None,
        help="출력 텍스트 파일 경로 (기본값: 출력 안함)"
    )
    parser.add_argument(
        "--use-date-filter",
        action="store_true",
        help="날짜 필터링 사용 (기본값: False, v5 subset 전체 사용)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="release_v5",
        help="사용할 subset (기본값: release_v5, 880개)"
    )
    
    args = parser.parse_args()
    
    debug_livecodebench_qwen_samples(
        num_samples=args.num_samples,
        output_file=args.output_file,
        yaml_path=args.yaml,
        use_date_filter=args.use_date_filter,
        subset=args.subset
    )

