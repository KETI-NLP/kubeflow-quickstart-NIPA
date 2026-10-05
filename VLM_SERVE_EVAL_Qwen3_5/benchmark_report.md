# VLM Benchmark Results

The following report displays the evaluation results of the newly integrated 5 VLM benchmarks, utilizing models `gpt-5.4` and `gemini-3.1-pro-preview`.

## Model: litellm_gpt5_4

### Task: mathvista_gen

#### Metrics
```json
{
  "mathvista_gen:default|0": {
    "extractive_match": 0.8,
    "extractive_match_stderr": 0.20000000000000004
  },
  "all": {
    "extractive_match": 0.8,
    "extractive_match_stderr": 0.20000000000000004
  }
}
```

#### Examples

**Example 1**

**Prompt:**
```
Look at the image carefully and answer the following question.

Approximately, what percentage of jewelry sales in January were Rings?
A. Around 21%
B. Around 27%
C. Around 31%
D. Around 37%

Respond with only the letter of the correct answer (e.g. 'Answer: A').
Answer:
```

**Ground Truth:**
```
 C
```

**Prediction:**
```
['Answer: C']
```

**Example 2**

**Prompt:**
```
Look at the image carefully and answer the following question concisely.

The owner of a bed and breakfast inn recalled how many guests the inn had hosted each day. What is the median of the numbers?'
Answer:
```

**Ground Truth:**
```
5
```

**Prediction:**
```
['The numbers are 4, 4, 5, 7, 5.\n\nSorted: 4, 4, 5, 5, 7\n\nThe median is **5**.']
```

### Task: ai2d_gen

#### Metrics
```json
{
  "ai2d_gen:default|0": {
    "extractive_match": 0.0
  },
  "all": {
    "extractive_match": 0.0
  }
}
```

#### Examples

**Example 1**

**Prompt:**
```
Look at the diagram carefully and answer the question.

What is at the top of the food chain?
A. mackerel
B. phytoplankton
C. small fish
D. large shark

Respond with only the letter of the correct answer (e.g. 'Answer: A').
Answer:
```

**Ground Truth:**
```
 A
```

**Prediction:**
```
['Answer: D']
```

### Task: chartqa_gen

#### Metrics
```json
{
  "chartqa_gen:default|0": {
    "em": 1.0
  },
  "all": {
    "em": 1.0
  }
}
```

#### Examples

**Example 1**

**Prompt:**
```
Look at the chart carefully and answer the following question concisely.
For numerical answers, give the exact number shown in the chart (e.g. '42' or '3.5').

How many app publishers were in Apple's App Store in 2017?
Answer:
```

**Ground Truth:**
```
143
```

**Prediction:**
```
['143']
```

### Task: hallusionbench_gen

#### Metrics
```json
{
  "hallusionbench_gen:default|0": {
    "extractive_match": 0.0,
    "extractive_match_stderr": 0.0
  },
  "all": {
    "extractive_match": 0.0,
    "extractive_match_stderr": 0.0
  }
}
```

#### Examples

**Example 1**

**Prompt:**
```
Look at the image carefully and answer with only 'Yes' or 'No'.

According to the lower text given in the image, is this poster for the DDT band?
Answer:
```

**Ground Truth:**
```
 No
```

**Prediction:**
```
['Yes']
```

**Example 2**

**Prompt:**
```
Look at the image carefully and answer with only 'Yes' or 'No'.

Are blue lines in the image parallel? Yes or No
Answer:
```

**Ground Truth:**
```
 No
```

**Prediction:**
```
['Yes']
```

### Task: scienceqa_gen

#### Metrics
```json
{
  "scienceqa_gen:default|0": {
    "extractive_match": 1.0,
    "extractive_match_stderr": 0.0
  },
  "all": {
    "extractive_match": 1.0,
    "extractive_match_stderr": 0.0
  }
}
```

#### Examples

**Example 1**

**Prompt:**
```
Hint: Select the better estimate.

Look at the image carefully and answer the science question.

What is the volume of a bottle of nail polish?
A. 15 milliliters
B. 15 liters

Respond with only the letter of the correct answer (e.g. 'Answer: A').
Answer:
```

**Ground Truth:**
```
 A
```

**Prediction:**
```
['Answer: A']
```

**Example 2**

**Prompt:**
```
Look at the image carefully and answer the science question.

Which change best matches the sentence?
The floor of a valley is covered by water after a heavy rainfall.
A. wildfire
B. flood
C. volcanic eruption

Respond with only the letter of the correct answer (e.g. 'Answer: A').
Answer:
```

**Ground Truth:**
```
 B
```

**Prediction:**
```
['Answer: B']
```

## Model: litellm_gemini_3_1_pro

### Task: mathvista_gen

#### Metrics
```json
{
  "mathvista_gen:default|0": {
    "extractive_match": 0.8,
    "extractive_match_stderr": 0.20000000000000004
  },
  "all": {
    "extractive_match": 0.8,
    "extractive_match_stderr": 0.20000000000000004
  }
}
```

#### Examples

**Example 1**

**Prompt:**
```
Look at the image carefully and answer the following question.

Approximately, what percentage of jewelry sales in January were Rings?
A. Around 21%
B. Around 27%
C. Around 31%
D. Around 37%

Respond with only the letter of the correct answer (e.g. 'Answer: A').
Answer:
```

**Ground Truth:**
```
 C
```

**Prediction:**
```
['Answer: C']
```

**Example 2**

**Prompt:**
```
Look at the image carefully and answer the following question concisely.

The owner of a bed and breakfast inn recalled how many guests the inn had hosted each day. What is the median of the numbers?'
Answer:
```

**Ground Truth:**
```
5
```

**Prediction:**
```
['Based on the table, the numbers of guests are 4, 4, 5, 7, and 5.\n\nTo find the median, first order the numbers from least to greatest:\n4, 4, 5, 5, 7\n\nThe median is the middle number in the ordered list, which is **5**.']
```

### Task: ai2d_gen

#### Metrics
```json
{
  "ai2d_gen:default|0": {
    "extractive_match": 0.0
  },
  "all": {
    "extractive_match": 0.0
  }
}
```

#### Examples

**Example 1**

**Prompt:**
```
Look at the diagram carefully and answer the question.

What is at the top of the food chain?
A. mackerel
B. phytoplankton
C. small fish
D. large shark

Respond with only the letter of the correct answer (e.g. 'Answer: A').
Answer:
```

**Ground Truth:**
```
 A
```

**Prediction:**
```
['Answer: D']
```

### Task: chartqa_gen

#### Metrics
```json
{
  "chartqa_gen:default|0": {
    "em": 1.0
  },
  "all": {
    "em": 1.0
  }
}
```

#### Examples

**Example 1**

**Prompt:**
```
Look at the chart carefully and answer the following question concisely.
For numerical answers, give the exact number shown in the chart (e.g. '42' or '3.5').

How much did the insurance industry spend on lobbying in 2020?
Answer:
```

**Ground Truth:**
```
151.85
```

**Prediction:**
```
['151.85']
```

**Example 2**

**Prompt:**
```
Look at the chart carefully and answer the following question concisely.
For numerical answers, give the exact number shown in the chart (e.g. '42' or '3.5').

Is the sum value of Africa and Americas more then Asia and Oceania?
Answer:
```

**Ground Truth:**
```
Yes
```

**Prediction:**
```
['Yes']
```

### Task: hallusionbench_gen

#### Metrics
```json
{
  "hallusionbench_gen:default|0": {
    "extractive_match": 0.0,
    "extractive_match_stderr": 0.0
  },
  "all": {
    "extractive_match": 0.0,
    "extractive_match_stderr": 0.0
  }
}
```

#### Examples

**Example 1**

**Prompt:**
```
Look at the image carefully and answer with only 'Yes' or 'No'.

According to the lower text given in the image, is this poster for the DDT band?
Answer:
```

**Ground Truth:**
```
 No
```

**Prediction:**
```
['Yes']
```

**Example 2**

**Prompt:**
```
Look at the image carefully and answer with only 'Yes' or 'No'.

Are blue lines in the image parallel? Yes or No
Answer:
```

**Ground Truth:**
```
 No
```

**Prediction:**
```
['Yes']
```

### Task: scienceqa_gen

#### Metrics
```json
{
  "scienceqa_gen:default|0": {
    "extractive_match": 1.0,
    "extractive_match_stderr": 0.0
  },
  "all": {
    "extractive_match": 1.0,
    "extractive_match_stderr": 0.0
  }
}
```

#### Examples

**Example 1**

**Prompt:**
```
Hint: The objects are identical except for their temperatures.

Look at the image carefully and answer the science question.

Which object has the least thermal energy?
A. a 200-gram cup of black tea at a temperature of 187°F
B. a 200-gram cup of black tea at a temperature of 154°F
C. a 200-gram cup of black tea at a temperature of 172°F

Respond with only the letter of the correct answer (e.g. 'Answer: A').
Answer:
```

**Ground Truth:**
```
 B
```

**Prediction:**
```
['Answer: B']
```

