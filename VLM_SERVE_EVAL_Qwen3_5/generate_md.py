import base64
from io import BytesIO
from datasets import load_from_disk
import json

def image_to_base64(img):
    try:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        return f"Error converting image: {e}"

def generate_sample_md(dataset_path, output_md_path):
    print(f"Loading dataset from {dataset_path}...")
    ds = load_from_disk(dataset_path)
    
    sample = ds['train'][0]
    
    md_content = f"# Dataset Sample from {dataset_path}\n\n"
    md_content += "## Keys in sample\n"
    md_content += f"`{list(sample.keys())}`\n\n"
    
    for key, value in sample.items():
        md_content += f"### {key}\n"
        if hasattr(value, 'mode') and hasattr(value, 'size'):  # likely PIL Image
            b64_img = image_to_base64(value)
            md_content += f"![{key}]({b64_img})\n\n"
        elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'mode'):
            # List of images
            for i, img in enumerate(value):
                b64_img = image_to_base64(img)
                md_content += f"**Image {i+1}**:\n\n![{key}_{i}]({b64_img})\n\n"
        elif isinstance(value, (dict, list)):
            md_content += f"```json\n{json.dumps(value, ensure_ascii=False, indent=2)}\n```\n\n"
        else:
            md_content += f"{value}\n\n"
            
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Successfully wrote sample to {output_md_path}")

if __name__ == "__main__":
    dataset_path = "/workspace/2026_llm_data_generation/llm_training_ready/data_korean_heritage_vqa_hf/"
    output_md_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/korean_heritage_vqa_sample.md"
    generate_sample_md(dataset_path, output_md_path)
