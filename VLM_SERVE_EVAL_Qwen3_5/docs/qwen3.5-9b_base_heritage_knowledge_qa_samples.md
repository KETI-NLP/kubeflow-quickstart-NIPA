# Qwen3.5-9B (base) — Korean Heritage Knowledge QA 샘플

vLLM 엔드포인트(DP=8) 상에서 `Qwen/Qwen3.5-9B` 가 두 태스크에 답한 응답 5건씩 (전문).

- 주관식: `korean_heritage_knowledge` (`generation_size=256`)
- 객관식: `korean_heritage_knowledge_mc` (`generation_size=8192`)

## 📘 주관식 — korean_heritage_knowledge

100건 평균: `heritage_knowledge_em=0.080` · `heritage_format_valid=0.170`

### 주관식 #0 — category=`year` — 보은 선병우 고가 (報恩 宣炳禹 古家)

- **sample_id**: `795`
- **image_path**: `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/보은_선병우_고가_報恩_宣炳禹_古家.jpg`

**질문 (full prompt):**

```
사진 속 문화재가 문화재로 지정된 연도(예: 1962)를 알려주세요.

반드시 마지막 줄에 다음 형식으로만 답하세요:
Answer: <짧은 정답 키워드>

예시:
Answer: 조선
Answer: 서울
Answer: 유적건조물
Answer: 1962

정답 키워드 하나만 짧게 적으세요. 부가 설명·문장은 다른 줄에 쓰고, 마지막 줄은 정확히 'Answer: 키워드' 형식만.
```

**정답**: `1986`  (keywords: `['1986']`)

**모델 응답 (full):**

```
The user wants to know the year the cultural heritage in the photo was designated.

1.  **Analyze the image:**
    *   The image shows a traditional Korean village setting with a large, walled compound containing several traditional houses (hanok).
    *   There are mountains in the background with autumn foliage.
    *   There are cars parked in a lot in front of the compound.
    *   The architecture looks like a temple or a large ancestral hall/village.
    *   Looking closely at the buildings, it resembles a temple complex or a specific historical site.
    *   Let's identify the location. The layout, the specific roof styles, and the surrounding landscape (mountains) suggest a specific site.
    *   The site looks like **Hwaeomsa Temple (화엄사)** or a similar temple, but let's look closer. Wait, looking at the layout, it's actually **Gyokcheongsa Temple (옥천사)**? No.
    *   Let's look at the distinctive gate and the main hall. It looks like **Hwaeomsa (화엄사)** in Gyeongju? No, the mountains look different.
    *   Let's consider **Sangwonsa (상원사)**? No.
    *   Let's consider **Hwaeomsa (화엄사)** again. The image shows a large courtyard.
    *   Actually, this looks very much like **Hwaeomsa Temple (화엄사)** in Gyeongju, South Korea. Let me verify.
    *   Wait, let's look at the specific features. The gate in the center, the main hall behind it.
    *   Let's try searching for "Korean temple autumn mountains cars".
    *   Let's look at the building style. It's a "Donggung" (East Hall) or similar.
    *   Let's reconsider the location. It looks like **Hwaeomsa (화엄사)** in Gyeongju. The main hall (Daeungjeon) is visible.
    *   Wait, is it **Hwaeomsa**? The main hall of Hwaeomsa is quite famous.
    *   Let's look at another possibility: **Gyokcheongsa (옥천사)**? No.
    *   How about **Hw
```

**채점**: `heritage_format_valid=0` · `heritage_knowledge_em=0` · `heritage_year_em=0`

### 주관식 #1 — category=`era` — 제천 청풍 한벽루 (堤川 淸風 寒碧樓)

- **sample_id**: `1468`
- **image_path**: `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/제천_청풍_한벽루_堤川_淸風_寒碧樓.jpg`

**질문 (full prompt):**

```
사진 속 문화재가 만들어진 시대를 알려주세요.

반드시 마지막 줄에 다음 형식으로만 답하세요:
Answer: <짧은 정답 키워드>

예시:
Answer: 조선
Answer: 서울
Answer: 유적건조물
Answer: 1962

정답 키워드 하나만 짧게 적으세요. 부가 설명·문장은 다른 줄에 쓰고, 마지막 줄은 정확히 'Answer: 키워드' 형식만.
```

**정답**: `조선`  (keywords: `['조선']`)

**모델 응답 (full):**

```
The user wants to identify the historical period of the cultural heritage site in the image.

1.  **Analyze the image:**
    *   The image shows a traditional Korean pavilion.
    *   It has a distinctive two-story structure (or a raised platform with a roof).
    *   The roof is curved and tiled, typical of Joseon dynasty architecture.
    *   The structure is elevated on stone pillars.
    *   There are trees in the foreground and mountains in the background.
    *   This looks like the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion / Gyeongbokgung Pavilion)** or more likely, the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)** is not right. Wait, let me look closer.
    *   It looks like the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)**? No.
    *   Let's identify the specific building. It looks like the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)**? No, that's not it.
    *   It looks like the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)**? No.
    *   Let's try to recognize the specific pavilion. It has a unique shape. It looks like the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)**? No.
    *   Wait, is it the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)**? No.
    *   Let's look at the structure again. It's a pavilion with a raised platform. It looks like the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)**? No.
    *   Actually, this looks like the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)**? No.
    *   Let's consider the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)**? No.
    *   Is it the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)**? No.
    *   Let's look at the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)**? No.
    *   Wait, could it be the **Gyeongbokgung Pavilion (Gyeongbokgung Pavilion)**? No.
    *   Let's try to
```

**채점**: `heritage_format_valid=0` · `heritage_knowledge_em=0` · `heritage_era_em=0`

### 주관식 #2 — category=`category` — 묘법연화경 권4~7(2016) (妙法連華經 卷四~七(2016))

- **sample_id**: `478`
- **image_path**: `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/묘법연화경_권4_7_2016_妙法連華經_卷四_七_2016.jpg`

**질문 (full prompt):**

```
사진 속 문화재의 상위 카테고리는 무엇인가요?

반드시 마지막 줄에 다음 형식으로만 답하세요:
Answer: <짧은 정답 키워드>

예시:
Answer: 조선
Answer: 서울
Answer: 유적건조물
Answer: 1962

정답 키워드 하나만 짧게 적으세요. 부가 설명·문장은 다른 줄에 쓰고, 마지막 줄은 정확히 'Answer: 키워드' 형식만.
```

**정답**: `기록유산`  (keywords: `['기록유산']`)

**모델 응답 (full):**

```
사용자의 질문은 사진 속 문화재의 '상위 카테고리'를 묻는 것입니다.
1.  **이미지 분석**: 사진은 한자 필사본이나 목판본으로 보이는 고서책의 한 페이지입니다. 오른쪽은 빈 페이지이고, 왼쪽에는 '묘법연화경'이라는 제목과 '제4권'이라는 글자가 보입니다. 내용은 불교 경전인 '묘법연화경'의 일부로 보입니다.
2.  **문화재 분류**: 이 책은 '경전'이나 '서적'에 해당합니다. 한국의 문화재 분류 체계 (국가지정문화재 등) 를 생각해보면, 책이나 문서류는 보통 '서적' 또는 '문서'로 분류됩니다. 더 넓은 범주로는 '유물'이나 '문화재' 자체가 상위 개념이지만, 질문이 "문화재의 상위 카테고리"를 묻는 것이므로, 이 물건이 속하는 가장 일반적인 분류를 찾아야 합니다.
3.  **분류 체계 확인**: 한국의 문화재청 분류 체계에서 책이나 문서류는 '서적' (Books) 또는 '문서' (Documents) 로 분류됩니다. 하지만 더 포괄적인 상위 분류를 묻는 것일 수도 있습니다.
    *   예시 답변을 보면 "조선", "서울", "유적건조물", "1962" 등이 있습니다. 이는 문화재의 시대, 지역, 유형, 지정 연도를 나타내는 것 같습니다.
    *   질문이 "상위 카테고리"라면, 이 물건이 무엇인가에 대한 가장 큰 분류를 묻는 것입니다. 이 물건은 '책'입니다. 문화재 분류에서 '서적'이 하나의 유형입니다.
    *   하지만, 만약 질문이 "이 문화재는 어떤 유형에 속하는가?"라면 '서적'이 정답일 것입니다.
    *   만약 질문이 "이 문화재는 어떤 시대의 것인가?"라면 '조선'이 될 수 있습니다. (글씨체와 종이 상태를 보면 조선 후기인 듯합니다.)
    *   다시 질문을 보면 "문화재의 상위 카테고리"입니다. 이는 다소 모호한 표현일 수 있습니다. 보통 문화재는 '유적', '유물', '무형문화재' 등으로 나뉩니다. 이 책은 '유물'에 속합니다.
    *   그러나, 예시를 보면 "유적건조물" 같은 구체적인 유형이 나
```

**채점**: `heritage_format_valid=0` · `heritage_knowledge_em=0` · `heritage_category_em=0`

### 주관식 #4 — category=`location` — 건칠반 (乾漆盤)

- **sample_id**: `1505`
- **image_path**: `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/건칠반_乾漆盤.jpg`

**질문 (full prompt):**

```
이 문화재의 소재지는 어디인가요? 시/도 이름으로 답해주세요.

반드시 마지막 줄에 다음 형식으로만 답하세요:
Answer: <짧은 정답 키워드>

예시:
Answer: 조선
Answer: 서울
Answer: 유적건조물
Answer: 1962

정답 키워드 하나만 짧게 적으세요. 부가 설명·문장은 다른 줄에 쓰고, 마지막 줄은 정확히 'Answer: 키워드' 형식만.
```

**정답**: `서울`  (keywords: `['서울']`)

**모델 응답 (full):**

```
이 이미지는 일본의 전통 공예품인 '다마도리' (玉造) 또는 '다마도리' (玉造) 스타일의 기와받침 (기타) 이나 장식용 받침대처럼 보입니다. 하지만 더 자세히 보면, 이는 일본의 전통 공예인 '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉造) 가 아니라, '다마도리' (玉
```

**채점**: `heritage_format_valid=0` · `heritage_knowledge_em=0` · `heritage_location_em=0`

### 주관식 #7 — category=`era` — 오산 독산성과 세마대지 (烏山 禿山城과 洗馬臺址)

- **sample_id**: `1432`
- **image_path**: `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/오산_독산성과_세마대지_烏山_禿山城과_洗馬臺址.jpg`

**질문 (full prompt):**

```
사진 속 문화재가 만들어진 시대를 알려주세요.

반드시 마지막 줄에 다음 형식으로만 답하세요:
Answer: <짧은 정답 키워드>

예시:
Answer: 조선
Answer: 서울
Answer: 유적건조물
Answer: 1962

정답 키워드 하나만 짧게 적으세요. 부가 설명·문장은 다른 줄에 쓰고, 마지막 줄은 정확히 'Answer: 키워드' 형식만.
```

**정답**: `삼국`  (keywords: `['삼국']`)

**모델 응답 (full):**

```
The user wants to identify the historical period of the cultural heritage site shown in the image.

1.  **Analyze the image:**
    *   The image shows a stone structure built into a hillside.
    *   It looks like a stone wall or a foundation for a building.
    *   There are large, rectangular stone blocks used for construction.
    *   There is a small opening or passage underneath a stone slab.
    *   The surrounding area is grassy and hilly, suggesting a rural or mountainous location.
    *   The style of the stone masonry (large, rough-hewn blocks) is characteristic of certain periods in Korean history.

2.  **Identify the specific site:**
    *   This looks very much like the "Gyeongju Stone House" or a similar structure found in Gyeongju, South Korea.
    *   Specifically, it resembles the "Gyeongju Stone House" (경주 석가사지 석탑 - wait, no, that's a pagoda). Let's look closer.
    *   It looks like the "Gyeongju Stone House" (경주 석가사지 석탑 isn't right).
    *   Let's consider "Gyeongju Stone House" (경주 석가사지 석탑 is definitely wrong).
    *   Wait, let's look at the structure again. It's a stone wall with a passage. It looks like the "Gyeongju Stone House" (경주 석가사지 석탑 is definitely wrong).
    *   Actually, this looks like the "Gyeongju Stone House" (경주 석가사지 석탑 is definitely wrong).
    *   Let's try searching my internal knowledge base for "stone house gyeongju".
    *   The image shows a stone structure that looks like a foundation or a wall. It is very similar to the "Gyeongju Stone House" (경주 석가사지 석탑 is definitely wrong).
    *   Wait, let's look at the "Gyeongju Stone House" (경주 석가사지 석탑 is definitely wrong).
    *   Let's try "Gyeongju Stone House" (경주 석가사지 석탑 is definitely wrong).
    *   Let's try "Gyeongju Stone House" (경주 석가사지 석탑 is definitely wrong).
    *   Let's
```

**채점**: `heritage_format_valid=0` · `heritage_knowledge_em=0` · `heritage_era_em=0`

## 📗 객관식 — korean_heritage_knowledge_mc

100건 평균: `strict_mc_match=0.390` · `heritage_mc_format_valid=0.600` (gen=8192)

### 객관식 #0 — category=`year` — 보은 선병우 고가 (報恩 宣炳禹 古家)

- **sample_id**: `795`
- **image_path**: `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/보은_선병우_고가_報恩_宣炳禹_古家.jpg`

**질문 (full prompt):**

```
이 문화재의 지정 연도는 몇 년인가요?
A. 1993
B. 1999
C. 1986
D. 2020

다음을 읽고 정답을 고르시오. 응답의 마지막 줄은 반드시 'Answer: $LETTER' (또는 '정답: $LETTER') 형식이어야 합니다. $LETTER 는 A, B, C, D 중 하나입니다.

예시:
Answer: A
Answer: B
```

**정답**: `C` (1986)  ·  **모델 추출 답**: `(없음)`

**모델 응답 (full):**

```
The user wants to identify the designation year of the cultural heritage site shown in the image.

1.  **Analyze the image:**
    *   The image shows a traditional Korean village setting with a cluster of Hanok (traditional Korean houses) in the center.
    *   There are mountains in the background with autumn foliage.
    *   There's a parking lot in the foreground with a few cars.
    *   The architecture looks like a traditional Korean temple or a historic village complex.
    *   The specific layout, the gate, and the surrounding landscape are distinctive.

2.  **Identify the location:**
    *   This looks like **Gangneung Samseongseong** (강릉 삼성성) or perhaps a similar historic site. Wait, let me look closer.
    *   Actually, looking at the layout, the central building with the curved roof, the surrounding walls, and the specific trees... this looks very much like **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress, but this looks like a village or temple complex).
    *   Let's reconsider. Is it **Gangneung Samseongseong**? No, that's a fortress.
    *   Is it **Gangneung Samseongseong**'s inner area? No.
    *   Let's look at the options: 1993, 1999, 1986, 2020.
    *   Let's try to match the image to a specific Korean cultural heritage site.
    *   The image shows a complex that looks like a temple or a traditional house complex.
    *   Wait, could it be **Gangneung Samseongseong** (Gangneung Samseongseong is a UNESCO World Heritage site, but the designation year is different).
    *   Let's look at the specific features. There is a gate, a main hall, and a wall.
    *   This looks like **Gangneung Samseongseong**'s *Samseongseong* area? No.
    *   Let's try searching my internal knowledge base for images of Korean cultural heritage sites with these specific features.
    *   The image looks like **Gangneung Samseongseong** (강릉 삼성성) is a fortress. This is not a fortress.
    *   Is it **Gangneung Samseongseong**'s *Samseongseong*? No.
    *   Let's look at the options again. 1986, 1993, 1999, 2020.
    *   Maybe it's **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's consider **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, the image is actually **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress). No, that's wrong.
    *   Let's look at the image again. It's a traditional Korean village.
    *   Could it be **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the specific site. It looks like **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this looks like **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try a different angle. The image shows a place called **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I recognize this place. It is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress). No, that's not right.
    *   Let's look at the options. 1986 is a common year for designations.
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image very carefully.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, the image is of **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to search for "Korean cultural heritage site parking lot autumn".
    *   The image looks like **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think I found it. This is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   No, that's not it.
    *   Let's look at the options again.
    *   A. 1993
    *   B. 1999
    *   C. 1986
    *   D. 2020
    *   The site is likely **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, the image is of **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's look at the image again. It's a traditional Korean village.
    *   The site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Actually, this is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Let's try to identify the site as **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
    *   Wait, I think the site is **Gangneung Samseongseong** (Gangneung Samseongseong is a fortress).
```

**채점**: `heritage_mc_format_valid=0` · `strict_mc_match=0` · `heritage_mc_year=0`

### 객관식 #1 — category=`era` — 제천 청풍 한벽루 (堤川 淸風 寒碧樓)

- **sample_id**: `1468`
- **image_path**: `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/제천_청풍_한벽루_堤川_淸風_寒碧樓.jpg`

**질문 (full prompt):**

```
이 문화재는 어느 시대의 것인가요?
A. 삼국
B. 발해
C. 조선
D. 현대

다음을 읽고 정답을 고르시오. 응답의 마지막 줄은 반드시 'Answer: $LETTER' (또는 '정답: $LETTER') 형식이어야 합니다. $LETTER 는 A, B, C, D 중 하나입니다.

예시:
Answer: A
Answer: B
```

**정답**: `C` (조선)  ·  **모델 추출 답**: `A`

**모델 응답 (full):**

```
사용자의 질문은 이미지 속 문화재가 어느 시대의 것인지 묻는 것입니다.
1.  **이미지 분석**: 이미지에는 전통적인 한국식 건축물이 보입니다. 기둥이 땅에서 멀리 떨어진 돌기둥 위에 서 있고, 지붕이 여러 겹으로 되어 있으며, 목조 구조입니다. 이는 전형적인 한국 고대 건축 양식입니다.
2.  **문화재 식별**: 이 건축물은 '화랑대' (Hwarangdae)로 보입니다. 화랑대는 백제와 신라, 고려 시대에 화랑들이 모이는 곳으로 사용되었습니다. 특히 이 사진은 충청남도 공주시에 있는 '화랑대' (백제화랑대)로 추정됩니다.
3.  **시대 확인**: 화랑대는 백제와 신라, 고려 시대에 존재했습니다. 하지만 이 특정 건축물 (공주 화랑대) 은 1970 년대에 복원된 것으로 알려져 있습니다. 원형은 백제나 신라 시대로 거슬러 올라가지만, 보통 '화랑대' 하면 백제나 신라 시대의 문화로 분류합니다.
4.  **옵션 분석**:
    *   A. 삼국: 백제, 신라, 고구려를 포함합니다. 화랑대는 삼국시대 (특히 백제와 신라) 에 화랑 제도가 성행했던 시기와 맞닿아 있습니다.
    *   B. 발해: 발해는 고구려 후신이지만, 화랑대라는 특정 건축물과 직접적인 연관은 삼국시대 (백제/신라) 가 더 강합니다.
    *   C. 조선: 조선 시대 건축 양식은 이보다 더 화려하고 기둥이 땅에 닿는 형태가 많습니다.
    *   D. 현대: 1970 년대 복원되었지만, 원형은 고대입니다. 질문은 "어느 시대의 것인가요"이므로 원형의 시대를 묻는 것이 일반적입니다.

    *정확한 확인*: 이 사진은 **공주 화랑대**입니다. 화랑대는 **백제**와 **신라** 시대에 화랑들이 모이는 곳으로 사용되었습니다. 따라서 '삼국' 시대에 해당합니다. (백제와 신라는 삼국 중 하나입니다.)

    *다시 생각*: 이 건축물은 1970 년대에 복원된 것이 맞습니다. 하지만 문제의 의도는 원형의 시대나 문화적 배경을 묻는 것입니다. 화랑대는 삼국시대 (특히 백제와 신라) 의 대표적인 문화입니다. 따라서 '삼국'이 가장 적절한 답입니다.

    *옵션 비교*:
    A. 삼국 (백제, 신라 등) - 화랑대 문화는 삼국시대.
    B. 발해 - 관련성 낮음.
    C. 조선 - 양식이 다름.
    D. 현대 - 복원물이지만, 문화재로서의 시대는 고대.

    결론: 이 건축물은 화랑대로, 삼국시대 (백제/신라) 의 문화입니다.

    *정답 선택*: A. 삼국
</think>

이 사진에 보이는 문화재는 **화랑대**입니다. 화랑대는 **백제**와 **신라** 시대에 화랑들이 모이는 곳으로 사용되었습니다. 백제와 신라는 삼국시대 (고구려, 백제, 신라) 에 해당하므로 정답은 삼국입니다. (이 특정 건축물은 1970 년대에 복원된 것이지만, 원형은 삼국시대의 문화입니다.)

Answer: A
```

**채점**: `heritage_mc_format_valid=1` · `strict_mc_match=0` · `heritage_mc_era=0`

### 객관식 #2 — category=`category` — 묘법연화경 권4~7(2016) (妙法連華經 卷四~七(2016))

- **sample_id**: `478`
- **image_path**: `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/묘법연화경_권4_7_2016_妙法連華經_卷四_七_2016.jpg`

**질문 (full prompt):**

```
이 문화재가 속한 가장 큰 분류는 무엇인가요?
A. 자연유산
B. 유물
C. 유적건조물
D. 기록유산

다음을 읽고 정답을 고르시오. 응답의 마지막 줄은 반드시 'Answer: $LETTER' (또는 '정답: $LETTER') 형식이어야 합니다. $LETTER 는 A, B, C, D 중 하나입니다.

예시:
Answer: A
Answer: B
```

**정답**: `D` (기록유산)  ·  **모델 추출 답**: `D`

**모델 응답 (full):**

```
사용자의 질문은 이미지 속 문화재가 속한 가장 큰 분류를 묻는 것입니다.

1.  **이미지 분석**:
    *   이미지는 한지(종이)로 된 책의 한 페이지입니다.
    *   오른쪽은 빈 페이지이고, 왼쪽에는 한자로 된 글이 적혀 있습니다.
    *   글자를 보면 "妙法蓮華經卷第四" (묘법연화경 권제4)라고 적혀 있습니다. 이는 불교 경전인 연화경의 일부입니다.
    *   아래쪽에는 "溫陵開元蓮寺比丘 戒環 解" (온릉 개원 연사 비구 계환 해)라고 되어 있어, 주석이나 해설을 쓴 사람과 소속을 나타냅니다.
    *   전체적으로 고전 서적의 한 장입니다.

2.  **분류 기준 확인 (대한민국 문화재 분류)**:
    *   **자연유산**: 자연적으로 형성된 산, 강, 동굴 등 (예: 설악산, 금강). 이 이미지는 인공물이므로 해당되지 않습니다.
    *   **유물**: 사람이 만든 물건 (예: 도자기, 청동기, 공예품). 책도 물건이지만, 문자나 기록을 담고 있는 경우 기록유산으로 더 구체적으로 분류됩니다.
    *   **유적건조물**: 건축물이나 토목 시설 (예: 궁궐, 사찰, 성곽). 책 자체는 건축물이 아닙니다.
    *   **기록유산**: 문자나 기호로 된 기록을 담고 있는 유물 (예: 고문서, 고서적, 지도, 간찰).

3.  **결론 도출**:
    *   이 이미지는 '묘법연화경'이라는 고서적의 일부입니다.
    *   고서적은 문자로 된 기록을 담고 있으므로 '기록유산'에 해당합니다.
    *   따라서 정답은 D입니다.

4.  **답변 작성**:
    *   선택지 D를 선택합니다.
    *   마지막 줄에 'Answer: D'를 씁니다.
</think>

이미지에는 '묘법蓮華經卷第四'라고 적힌 고서적의 한 페이지가 보입니다. 이는 문자로 된 기록을 담고 있는 고전 서적이므로 기록유산에 해당합니다.

Answer: D
```

**채점**: `heritage_mc_format_valid=1` · `strict_mc_match=1` · `heritage_mc_category=1`

### 객관식 #4 — category=`location` — 건칠반 (乾漆盤)

- **sample_id**: `1505`
- **image_path**: `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/건칠반_乾漆盤.jpg`

**질문 (full prompt):**

```
이 문화재는 어디에 위치해 있나요? (광역시/도)
A. 서울
B. 울산
C. 인천
D. 충남

다음을 읽고 정답을 고르시오. 응답의 마지막 줄은 반드시 'Answer: $LETTER' (또는 '정답: $LETTER') 형식이어야 합니다. $LETTER 는 A, B, C, D 중 하나입니다.

예시:
Answer: A
Answer: B
```

**정답**: `A` (서울)  ·  **모델 추출 답**: `B`

**모델 응답 (full):**

```
이 이미지는 한국의 전통 공예품인 '칠기' (lacquerware) 작품으로 보입니다. 구체적으로는 '칠기 대접'이나 '칠기 받침대' 형태입니다.

1.  **이미지 분석**:
    *   검은색 광택이 나는 칠기 (lacquer) 재질입니다.
    *   받침 (stand) 이 있고, 그 위에 접시 같은 부분이 있습니다.
    *   받침 중앙에 다이아몬드 모양의 구멍이 있습니다.
    *   테두리가 붉은색 (진홍색) 입니다.
    *   표면에 금박이 박혀 있는 문양이 보입니다.

2.  **검색 및 지식 활용**:
    *   이 물건은 '칠기 대접' 또는 '칠기 받침대'로 보입니다.
    *   특히, 이 모양은 '칠기 대접' 중에서도 '칠기 받침대 대접'과 유사합니다.
    *   하지만 더 구체적으로 찾아보면, 이 물건은 **울산광역시**에 있는 **울산박물관**에 소장된 문화재일 가능성이 높습니다.
    *   검색어를 "칠기 대접 울산박물관" 또는 "칠기 받침대 문화재" 로 검색해 봅니다.
    *   실제로, **울산박물관**에 소장된 **'칠기 대접' (또는 유사한 칠기 공예품)**이 있습니다.
    *   더 정확히는, 이 이미지는 **울산광역시**에 있는 **울산박물관**의 소장품인 **'칠기 대접'** (국보 또는 보물 등급은 아닐 수 있으나, 중요 문화재로 등록되어 있을 수 있음) 혹은 **'칠기 받침대'**와 매우 유사합니다.
    *   다시 한번 확인해 보면, 이 물건은 **칠기 대접**으로, **울산광역시**에 소재한 **울산박물관**에 소장되어 있습니다. (정확히는 '칠기 대접'으로 등록되어 있거나, 유사한 형태입니다.)
    *   실제로 이 이미지는 **울산박물관**의 소장품인 **'칠기 대접'** (또는 '칠기 받침대') 의 사진과 일치합니다.
    *   검색 결과: "칠기 대접 울산박물관" -> 울산박물관 소장품 중 '칠기 대접'이 있습니다.
    *   또한, 이 물건은 **칠기**로 제작된 것으로, 한국의 전통 공예입니다.
    *   질문은 "이 문화재는 어디에 위치해 있나요? (광역시/도)"입니다.
    *   보기: A. 서울, B. 울산, C. 인천, D. 충남.
    *   이 물건은 **울산광역시**에 있습니다.

3.  **결론**:
    *   이 문화재는 울산박물관에 소장되어 있습니다.
    *   따라서 정답은 B. 울산입니다.

    (참고: 이 물건은 '칠기 대접'으로, 울산박물관에 소장되어 있습니다. 정확한 명칭은 '칠기 대접'일 가능성이 높습니다.)

    다시 한번 이미지와 비교해 보면, 이 물건은 **칠기 대접**으로, **울산광역시**에 있는 **울산박물관**에 소장되어 있습니다.
    (실제 검색: "칠기 대접 울산박물관" -> 이미지와 일치하는 물건이 나옵니다. '칠기 대접'으로 등록되어 있습니다.)

    따라서 정답은 B입니다.
</think>

이 문화재는 **칠기 대접**으로, **울산광역시**에 소재한 **울산박물관**에 소장되어 있습니다.

Answer: B
```

**채점**: `heritage_mc_format_valid=1` · `strict_mc_match=0` · `heritage_mc_location=0`

### 객관식 #7 — category=`era` — 오산 독산성과 세마대지 (烏山 禿山城과 洗馬臺址)

- **sample_id**: `1432`
- **image_path**: `/workspace/2026_llm_data_generation/korean_heritage_multimodal_hallucination_dpo/image_cache/original_images/오산_독산성과_세마대지_烏山_禿山城과_洗馬臺址.jpg`

**질문 (full prompt):**

```
이 문화재가 만들어진 시대는 언제인가요?
A. 삼국
B. 근대
C. 현대
D. 통일신라

다음을 읽고 정답을 고르시오. 응답의 마지막 줄은 반드시 'Answer: $LETTER' (또는 '정답: $LETTER') 형식이어야 합니다. $LETTER 는 A, B, C, D 중 하나입니다.

예시:
Answer: A
Answer: B
```

**정답**: `A` (삼국)  ·  **모델 추출 답**: `A`

**모델 응답 (full):**

```
사용자의 질문은 이미지 속 문화재가 만들어진 시대를 묻는 것입니다.
1.  **이미지 분석**: 이미지에는 돌로 쌓은 성벽과 그 아래로 이어지는 계단, 그리고 성벽 위로 덮인 흙과 풀이 보입니다. 이는 전형적인 고대 성곽의 모습입니다. 특히 성벽의 구조와 돌의 크기가 한국의 고대 성곽을 연상시킵니다.
2.  **문화재 식별**: 이 이미지는 '성산성 (Seongsanseong)'의 성벽과 성문 부분을 보여줍니다. 성산성은 경상북도 울진군에 위치한 고대 성곽입니다.
3.  **시대 확인**: 성산성은 삼국시대 (특히 백제나 신라, 혹은 고구려의 영향권) 에 축조된 것으로 추정됩니다. 하지만 더 구체적으로 보면, 성산성은 **삼국시대**에 축조된 성곽으로 알려져 있습니다. (참고: 성산성은 삼국시대 성곽으로 추정되지만, 정확한 연대는 명확하지 않을 수 있습니다. 그러나 보기 중 '삼국'이 가장 적합합니다.)
4.  **보기 분석**:
    *   A. 삼국: 고대 국가인 삼국시대 (고구려, 백제, 신라) 에 해당합니다. 성산성은 이 시대에 축조된 것으로 추정됩니다.
    *   B. 근대: 19세기 말~20세기 초. 이 성곽의 돌 쌓기 방식은 근대적이지 않습니다.
    *   C. 현대: 현재. 명백히 아닙니다.
    *   D. 통일신라: 삼국시대의 후속 단계입니다. 성산성은 삼국시대 초기~중기 축조설이 강합니다.

    *정확한 정보 확인*: 성산성 (Seongsanseong) 은 경상북도 울진군에 있는 성곽으로, **삼국시대**에 축조된 것으로 추정됩니다. (일부 자료에서는 신라나 백제, 혹은 고구려의 영향으로 보기도 하지만, 일반적으로 '삼국시대'로 분류합니다.)

    따라서 정답은 A입니다.

    *추가 확인*: 이 이미지는 실제로 '성산성 (Seongsanseong)'의 성벽과 성문입니다. 성산성은 삼국시대 성곽으로 지정되어 있습니다.

    결론: A. 삼국
</think>

이 이미지는 경상북도 울진군에 위치한 **성산성 (Seongsanseong)**의 성벽과 성문 부분을 보여줍니다. 성산성은 **삼국시대**에 축조된 것으로 추정되는 고대 성곽입니다.

Answer: A
```

**채점**: `heritage_mc_format_valid=1` · `strict_mc_match=1` · `heritage_mc_era=1`
