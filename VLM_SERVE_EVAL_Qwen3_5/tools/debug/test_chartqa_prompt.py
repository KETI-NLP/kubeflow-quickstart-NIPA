from custom_tasks.custom_chartqa_task import chartqa_prompt
from PIL import Image

line = {
    "question": "What is the correlation of user numbers with internet connectivity among seniors?",
    "answer": "none",
    "image": Image.new("RGB", (224, 224)),
    "type": "human"
}

doc = chartqa_prompt(line)
print("Doc query:", doc.query)
print("Doc images:", doc.images)
