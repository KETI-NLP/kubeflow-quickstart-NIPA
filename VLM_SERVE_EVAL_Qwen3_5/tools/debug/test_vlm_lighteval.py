from lighteval.models.transformers.vlm_transformers_model import VLMTransformersModelConfig, VLMTransformersModel
from lighteval.tasks.requests import Doc
from lighteval.models.abstract_model import GenerationParameters
from PIL import Image

config = VLMTransformersModelConfig(
    model_name="Qwen/Qwen3.5-9B",
    generation_parameters=GenerationParameters(temperature=0.0)
)
model = VLMTransformersModel(config=config)

doc = Doc(
    task_name="test",
    query="Look at the chart.",
    choices=["42"],
    gold_index=0,
    images=[Image.new('RGB', (10, 10))],
    instruction="",
    generation_size=10,
    num_samples=1,
    use_logits=False
)

print("Original query:", doc.query)
try:
    results = model.greedy_until([doc])
    print("Results:", results)
except Exception as e:
    import traceback
    traceback.print_exc()
