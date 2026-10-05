# GPT-4o Korean Heritage Name VQA Error Examples

실험 결과 파일:
- [results_2026-04-13T01-15-49.742810.json](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_name_vqa_gpt4o_smoke_v2/results/openai/gpt-4o/results_2026-04-13T01-15-49.742810.json)
- [details_korean_heritage_name_vqa|0_2026-04-13T01-15-49.742810.parquet](/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/test_korean_heritage_name_vqa_gpt4o_smoke_v2/details/openai/gpt-4o/2026-04-13T01-15-49.742810/details_korean_heritage_name_vqa%7C0_2026-04-13T01-15-49.742810.parquet)

요약:
- smoke run 20개 기준 `name_vqa_format_valid = 1.0`
- 하지만 `name_vqa_strict_em = 0.0`, `name_vqa_relaxed_em = 0.0`
- 즉, 형식은 잘 지켰지만 문화재 이름 자체를 자주 다른 유물명으로 오인했다.

## Example 1

- `sample_id`: `name_vqa_12837_name_05`
- 질문: `사진 속 한국 문화재가 무엇인지 이름만 답해 주세요.`
- 정답: `영원사목불좌상및복장유물`
- GPT-4o 답변: `Answer: 금동여래좌상`
- 판정: 완전 오답
- 이미지 URL: https://www.heritage.go.kr/unisearch/images/tangible_cult_prop/1638204.jpg
- [원본 이미지 파일](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/영원사목불좌상및복장유물_영원寺木佛坐像및腹藏遺物.jpg)

![Example 1](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/영원사목불좌상및복장유물_영원寺木佛坐像및腹藏遺物.jpg)

## Example 2

- `sample_id`: `name_vqa_03748_name_01`
- 질문: `이 이미지에 나온 한국 문화재의 이름은 무엇인가요? 문화재명만 답하세요.`
- 정답: `분천리까치구멍집`
- GPT-4o 답변: `Answer: 정선 아우라지 뱃사공 집`
- 판정: 완전 오답
- 이미지 URL: https://www.heritage.go.kr/unisearch/images/folklore_material/1658828.jpg
- [원본 이미지 파일](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/분천리까치구멍집_汾川里까치구멍집.jpg)

![Example 2](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/분천리까치구멍집_汾川里까치구멍집.jpg)

## Example 3

- `sample_id`: `name_vqa_11201_name_06`
- 질문: `이 문화재의 이름을 답하세요. 다른 설명은 쓰지 마세요.`
- 정답: `제천 입석리 선돌`
- GPT-4o 답변: `Answer: 영월 흥원창대기비`
- 판정: 완전 오답
- 이미지 URL: https://www.heritage.go.kr/unisearch/images/monument/1650753.jpg
- [원본 이미지 파일](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/제천_입석리_선돌_堤川_立石里_立石.jpg)

![Example 3](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/제천_입석리_선돌_堤川_立石里_立石.jpg)

## Example 4

- `sample_id`: `name_vqa_03623_name_08`
- 질문: `사진 속 대상의 문화재명을 간단히 답하세요. 문화재 이름 한 줄만 출력하세요.`
- 정답: `잠실뽕나무`
- GPT-4o 답변: `Answer: 서울 남대문로 회화나무`
- 판정: 완전 오답
- 이미지 URL: https://www.heritage.go.kr/unisearch/images/nature_heri/2024101510330800.jpg
- [원본 이미지 파일](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/잠실뽕나무_蠶室_뽕나무.jpg)

![Example 4](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/잠실뽕나무_蠶室_뽕나무.jpg)

## Example 5

- `sample_id`: `name_vqa_08020_name_02`
- 질문: `사진 속 문화재의 명칭을 적어 주세요. 설명 없이 이름만 출력하세요.`
- 정답: `남해용문사현왕탱`
- GPT-4o 답변: `Answer: 삼불제석도`
- 판정: 완전 오답
- 이미지 URL: https://www.heritage.go.kr/unisearch/images/cultural_material/1666109.jpg
- [원본 이미지 파일](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/남해용문사현왕탱_南海龍門寺現王幀.jpg)

![Example 5](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/남해용문사현왕탱_南海龍門寺現王幀.jpg)

## Example 6

- `sample_id`: `name_vqa_12480_name_02`
- 질문: `사진 속 문화재의 명칭을 적어 주세요. 설명 없이 이름만 출력하세요.`
- 정답: `초조본 아비담팔건도론 권24`
- GPT-4o 답변: `Answer: 무구정광대다라니경`
- 판정: 완전 오답
- 이미지 URL: https://www.heritage.go.kr/unisearch/images/treasure/1614460.jpg
- [원본 이미지 파일](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/초조본_아비담팔건도론_권24_初雕本_阿毗曇八揵度論_卷二十四.jpg)

![Example 6](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/초조본_아비담팔건도론_권24_初雕本_阿毗曇八揵度論_卷二十四.jpg)

## Example 7 Near Miss

- `sample_id`: `name_vqa_08363_name_01`
- 질문: `이 이미지에 나온 한국 문화재의 이름은 무엇인가요? 문화재명만 답하세요.`
- 정답: `구례 화엄사 각황전 앞 석등`
- GPT-4o 답변: `Answer: 화엄사 각황전 앞 석등`
- 판정: 현재 strict / relaxed EM 기준에서는 오답
- 메모: 실제로는 거의 맞췄지만 앞의 지명 `구례`가 빠져서 0점 처리되었다.
- 이미지 URL: https://www.heritage.go.kr/unisearch/images/national_treasure/1612398.jpg
- [원본 이미지 파일](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/구례_화엄사_각황전_앞_석등_求禮_華嚴寺_覺皇殿_앞_石燈.jpg)

![Example 7](/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/구례_화엄사_각황전_앞_석등_求禮_華嚴寺_覺皇殿_앞_石燈.jpg)

## Quick Reading

- 오답 다수는 같은 계열 문화재로의 치환이다.
- 불화는 다른 불화명으로, 석조물은 다른 석조물명으로, 고문서는 다른 고문서명으로 오인하는 경향이 보인다.
- 즉, 이미지 입력이 안 들어간 문제라기보다는 들어간 뒤에도 세밀한 문화재 식별이 어렵다는 쪽에 가깝다.
- 다만 `구례 화엄사 각황전 앞 석등 -> 화엄사 각황전 앞 석등`처럼 부분 일치성은 보이는 사례가 있어 alias/부분일치 metric을 추가하면 조금 더 진단적인 평가가 가능하다.
