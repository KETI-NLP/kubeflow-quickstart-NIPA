"""
MBPP 커스텀 테스크 디버깅 스크립트
"""

import os
import sys
import yaml
import json
import subprocess
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, TextIO

from datasets import load_dataset
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from litellm import completion

sys.path.insert(0, str(Path(__file__).parent.parent / "custom_tasks"))
from custom_mbpp_task import (
    create_custom_mbpp_task,
    MBPPCodegenMetric,
    mbpp_prompt,
    record_to_sample,
    extract_code,
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
        max_tokens = config.get('max_model_length') or gen_params.get('max_tokens', 15000)
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


def run_mbpp_test(code: str, test_list: list, test_setup_code: str = "", timeout: int = 6) -> dict: 
    if not test_list:
        return {'passed': False, 'error': 'No test_list provided', 'output': None}
    
    full_code = f"""{test_setup_code}

{code}

# Run tests - 에러 발생 시 즉시 종료 (returncode != 0)
"""
    for assert_stmt in test_list:
        full_code += f"{assert_stmt}\n"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        passed = (result.returncode == 0)
        error = None
        if not passed:
            if result.stderr:
                error = result.stderr.strip().split('\n')[-1]  
            elif result.stdout:
                error = result.stdout.strip().split('\n')[-1]
        
        return {
            'passed': passed,
            'error': error,
            'output': result.stdout if result.returncode == 0 else result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'passed': False,
            'error': f'Timeout after {timeout} seconds',
            'output': None,
            'returncode': -1
        }
    except Exception as e:
        return {
            'passed': False,
            'error': str(e),
            'output': None,
            'returncode': -1
        }
    finally:
        try:
            Path(temp_file).unlink()
        except:
            pass


def debug_mbpp_samples(
    subset: str = "full",
    split: str = "test",
    num_samples: int = 5,
    output_file: Optional[str] = None,
    yaml_path: str = None
):
    if yaml_path is None:
        script_dir = Path(__file__).parent
        yaml_path = str(script_dir / "litellm_qwen3.yaml")
    else:
        yaml_path = str(yaml_path)
    
    yaml_path_obj = Path(yaml_path)
    if not yaml_path_obj.exists():
        print(f"ERROR: LiteLLM 설정 파일을 찾을 수 없습니다: {yaml_path_obj}")
        return
    
    output_f = None
    original_stdout = None
    original_stderr = None
    if output_file:
        output_f = open(output_file, 'w', encoding='utf-8')
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeOutput(sys.stdout, output_f)
        sys.stderr = TeeOutput(sys.stderr, output_f)
    
    try:
        print("=" * 80)
        print("MBPP 커스텀 테스크 디버깅 스크립트")
        print("=" * 80)
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Subset: {subset}, Split: {split}")
        print(f"디버깅할 샘플 수: {num_samples}")
        if output_file:
            print(f"출력 파일: {output_file}")
        print()
        
        print("1. LiteLLM 설정 로드 중...")
        config = load_litellm_config(str(yaml_path))
        print(f"   모델: {config['model_name']}")
        print()
        
        print("2. 데이터셋 로드 중...")
        hf_subset = subset if subset != "full" else None
        dataset = load_dataset("google-research-datasets/mbpp", hf_subset, split=split)
        print(f"   데이터셋 크기: {len(dataset)}")
        print()
        
        print("3. 커스텀 태스크 생성 중...")
        task_config = create_custom_mbpp_task(subset=subset, split=split)
        print(f"   태스크 이름: {task_config.name}")
        print(f"   Few-shot 개수: {task_config.num_fewshots}")
        print(f"   Few-shot split: {task_config.few_shots_split}")
        print()
        
        print("4. LightevalTask 생성 중...")
        task = LightevalTask(config=task_config)
        print(f"   Few-shot pool 크기: {len(task.fewshot_docs())}")
        print()
        
        prompt_manager = PromptManager(use_chat_template=False)
        
        metric = MBPPCodegenMetric()
        
        print("5. 샘플 디버깅 시작...")
        print("=" * 80)
        
        total_passed = 0
        total_samples = min(num_samples, len(dataset))
        
        docs_with_fewshot = task.get_docs(max_samples=total_samples)
        
        for idx in range(total_samples):
            print(f"\n[샘플 {idx + 1}/{total_samples}]")
            print("-" * 80)
            
            record = dataset[idx]
            print(f"Task ID: {record.get('task_id', 'N/A')}")

            if idx < len(docs_with_fewshot):
                doc_with_fewshot = docs_with_fewshot[idx]
                doc = doc_with_fewshot  
                
                print(f"\n[Few-shot 예제] ({len(doc_with_fewshot.fewshot_samples)}개)")
                for i, fewshot_doc in enumerate(doc_with_fewshot.fewshot_samples, 1):
                    print(f"\n  예제 {i}:")
                    print(f"    문제: {fewshot_doc.query}")
                    if fewshot_doc.choices:
                        code_preview = fewshot_doc.choices[0]
                        if len(code_preview) > 200:
                            print(f"    코드: {code_preview[:200]}...")
                        else:
                            print(f"    코드: {code_preview}")
                print()
                
                actual_prompt = prompt_manager.prepare_prompt(doc_with_fewshot)
                print("[최종 프롬프트 (Few-shot 포함)]")
                print("=" * 80)
                print(actual_prompt)
                print("=" * 80)
                print()
                
                prompt_for_model = actual_prompt
            else:
                sample = record_to_sample(record)
                doc = mbpp_prompt(sample, task_name="mbpp_gen")
                print("\n[프롬프트 (Few-shot 없음)]")
                print(doc.query)
                print()
                prompt_for_model = doc.query
            
            test_list = doc.specific["test_list"]
            test_setup_code = doc.specific.get("test_setup_code", "")

            
            print("[모델 호출 중...]")
            model_output = call_model(prompt_for_model, config)
            print(f"[모델 출력 (길이: {len(model_output)} 문자)]")
            print(model_output)  
            print()
            
            extracted_code = extract_code(model_output, test_list)
            print(f"[추출된 코드 (길이: {len(extracted_code)} 문자)]")
            if extracted_code:
                print(extracted_code)  
            else:
                print("(코드 추출 실패)")
            print()


            print(f"[테스트 케이스] ({len(test_list)}개)")
            if test_setup_code:
                print(f"Setup Code:\n{test_setup_code}")
            for i, test in enumerate(test_list, 1):
                print(f"{test}")
            print()

            print("[테스트 실행 중...]")
            if test_setup_code:
                print(f"[Test Setup Code]:\n{test_setup_code}")
            
            print(f"[AssertionError 감지 테스트]")
            print(f"   테스트 케이스 수: {len(test_list)}")
            
            test_result = run_mbpp_test(
                extracted_code,
                test_list,
                test_setup_code,
                timeout=6
            )
            
            print(f"[테스트 실행 결과]")
            print(f"   Return code: {test_result.get('returncode', 'N/A')}")
            
            if test_result['passed']:
                print("✅ 테스트 통과!")
                total_passed += 1
                
                if test_result.get('returncode') == 0:
                    print("   ✓ AssertionError 감지 정상 (returncode == 0)")
                else:
                    print(f"   ⚠️ 경고: returncode가 0이 아닌데 passed=True입니다!")
            else:
                print("❌ 테스트 실패")
                if test_result.get('returncode') is not None:
                    if test_result['returncode'] != 0:
                        print(f"   ✓ AssertionError 감지 정상 (returncode != 0)")
                    else:
                        print(f"   ⚠️ 경고: returncode가 0인데 passed=False입니다!")
                if test_result['error']:
                    print(f"   에러: {test_result['error']}")
                if test_result['output']:
                    output_lines = test_result['output'].strip().split('\n')
                    if len(output_lines) > 5:
                        print(f"   출력 (마지막 5줄):")
                        for line in output_lines[-5:]:
                            print(f"     {line}")
                    else:
                        print(f"   출력:\n{test_result['output']}")
            print()
            
            model_response = ModelResponse(text=[model_output])
            try:
                metric_score = metric.compute(model_response, doc)
                print(f"[메트릭 점수]: {metric_score}")
            except Exception as e:
                print(f"[메트릭 계산 에러]: {e}")
            print()
        
        print("=" * 80)
        print(f"[최종 결과]")
        print(f"통과한 샘플: {total_passed}/{total_samples}")
        print(f"통과율: {total_passed/total_samples*100:.1f}%")
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if output_f:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except:
                pass
            if original_stdout:
                sys.stdout = original_stdout
            if original_stderr:
                sys.stderr = original_stderr
            output_f.close()
            if original_stdout:
                original_stdout.write(f"\n출력이 파일에 저장되었습니다: {output_file}\n")
                original_stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MBPP 커스텀 테스크 디버깅 스크립트")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="디버깅할 샘플 수 (기본값: 5)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="full",
        choices=["full", "sanitized"],
        help="MBPP 서브셋 (기본값: full)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "validation", "prompt"],
        help="데이터셋 스플릿 (기본값: test)"
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        default=None,
        help="출력 파일 경로 (선택사항)"
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="LiteLLM YAML 설정 파일 경로 (기본값: litellm_qwen3.yaml)"
    )
    
    args = parser.parse_args()
    
    debug_mbpp_samples(
        subset=args.subset,
        split=args.split,
        num_samples=args.num_samples,
        output_file=args.output_file,
        yaml_path=args.yaml
    )

