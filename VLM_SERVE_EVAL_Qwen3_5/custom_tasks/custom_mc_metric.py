import re
import numpy as np
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.tasks.requests import Doc, SamplingMethod

class StrictMultipleChoiceMatch(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        pred = ""
        if isinstance(model_response, list) and len(model_response) > 0:
            item = model_response[0]
        else:
            item = model_response
            
        if isinstance(item, dict):
            pred = item.get('text', '')
        elif hasattr(item, 'final_text'):
            if isinstance(item.final_text, list):
                pred = "".join(item.final_text)
            else:
                pred = item.final_text
        elif isinstance(item, str):
            pred = item
        else:
            pred = str(item)
            
        # If it's a free-form question (only 1 choice provided, which is the answer)
        if len(doc.choices) == 1:
            gold = doc.choices[0].strip().lower()
            pred_clean = pred.strip().lower()
            if gold in pred_clean:
                return 1.0
            try:
                gold_val = float(gold.replace(',', ''))
                pred_numbers = re.findall(r'-?\d+\.?\d*', pred_clean.replace(',', ''))
                for num_str in reversed(pred_numbers):
                    if abs(float(num_str) - gold_val) < 1e-5:
                        return 1.0
            except ValueError:
                pass
            return 0.0
            
        # Extract the target letter from choices (e.g. " A" -> "A")
        gold = doc.choices[doc.gold_index].strip().upper()
        
        # 1. Look for explicit pattern: Answer: X, 정답: X, 정답은 X
        # Find all text that comes after these keywords
        explicit_match = re.findall(r'(?i)(?:answer|정답|정답은)[\s]*:?[\s]*(.*)', pred)
        if explicit_match:
            # Take the very last occurrence (usually the final answer block)
            answer_text = explicit_match[-1].upper()
            
            # Find the first standalone A-E in that specific answer text
            matches = re.findall(r'\b([A-E])\b', answer_text)
            if matches:
                return 1.0 if matches[0] == gold else 0.0
            
            # If it explicitly gave an answer but there is no A-E (e.g. Answer: F)
            return 0.0
            
        # 2. Look for the letter exactly at the end of the text
        end_match = re.search(r'\b([A-E])\b[^\w]*$', pred.upper())
        if end_match:
            extracted = end_match.group(1).upper()
            return 1.0 if extracted == gold else 0.0

        # 3. Fallback: clean the text and find the last standalone letter
        clean_pred = re.sub(r'(?i)answer|정답|정답은', '', pred)
        matches = re.findall(r'\b([A-E])\b', clean_pred.upper())
        
        if not matches:
            matches = re.findall(r'[\(\[\s]([A-E])[\)\]\.\s]|^([A-E])[\.\)]', clean_pred.upper())
            matches = [m for tuple_match in matches for m in tuple_match if m]

        if matches:
            extracted = matches[-1]
            return 1.0 if extracted == gold else 0.0
            
        return 0.0

strict_mc_metric = SampleLevelMetric(
    metric_name="strict_mc_match",
    sample_level_fn=StrictMultipleChoiceMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
