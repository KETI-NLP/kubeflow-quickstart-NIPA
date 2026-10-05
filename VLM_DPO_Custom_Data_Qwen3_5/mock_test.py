import datasets
import json

ds = datasets.Dataset.from_list([
    {'prompt': [{'content': [{'type': 'image'}, {'type': 'text', 'text': 'Hello'}]}]},
    {'prompt': [{'content': [{'type': 'text', 'text': 'World'}]}]}
])

print("Original:")
print(json.dumps(ds[0], indent=2))

def map_fn(example):
    prompt = example["prompt"]
    # return exactly what extract_images_from_messages returned BEFORE my fix:
    new_prompt = []
    for message in prompt:
        new_content = []
        for item in message["content"]:
            if item["type"] == "image":
                new_content.append({"type": "image"})
            else:
                new_content.append({"type": "text", "text": item.get("text", "")})
        new_prompt.append({"role": "user", "content": new_content})
    return {"prompt": new_prompt}

mapped_ds = ds.map(map_fn)
print("\nMapped:")
print(json.dumps(mapped_ds[0], indent=2))
