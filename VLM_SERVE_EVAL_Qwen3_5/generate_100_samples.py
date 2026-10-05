import pyarrow as pa
import base64
import json
import random
from io import BytesIO
from PIL import Image

def image_to_base64(img_bytes):
    try:
        # Resize image to make base64 string smaller
        img = Image.open(BytesIO(img_bytes))
        img.thumbnail((256, 256))
        buffered = BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        return f"Error converting image: {e}"

def process_content(content_list):
    md = ""
    text_content = ""
    if hasattr(content_list, 'tolist'):
        content_list = content_list.tolist()
        
    for item in content_list:
        if isinstance(item, dict):
            if item.get('type') == 'image' and item.get('image'):
                img_dict = item['image']
                if 'bytes' in img_dict and img_dict['bytes']:
                    b64_img = image_to_base64(img_dict['bytes'])
                    path = img_dict.get('path', 'image')
                    md += f"**Image**: `{path}`\n\n![{path}]({b64_img})\n\n"
            elif item.get('type') == 'text' and item.get('text'):
                text = item['text']
                md += f"{text}\n\n"
                text_content += text
    return md, text_content

def process_messages(messages):
    md = ""
    full_text = ""
    if hasattr(messages, 'tolist'):
        messages = messages.tolist()
        
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown').capitalize()
            md += f"#### {role}:\n\n"
            if 'content' in msg:
                content_md, content_text = process_content(msg['content'])
                md += content_md
                full_text += content_text
    return md, full_text

def main():
    arrow_path = "/workspace/2026_llm_data_generation/llm_training_ready/data_korean_heritage_vqa_hf/train/data-00000-of-00486.arrow"
    output_md_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/korean_heritage_vqa_100_samples.md"
    
    samples_name = []
    samples_random = []
    
    print(f"Reading from {arrow_path}...")
    
    with pa.OSFile(arrow_path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        
        for batch in reader:
            df = batch.to_pandas()
            for i in range(len(df)):
                sample = df.iloc[i].to_dict()
                
                # Check prompt text
                _, prompt_text = process_messages(sample.get('prompt', []))
                
                is_name_query = "이름" in prompt_text or "명칭" in prompt_text
                
                if is_name_query and len(samples_name) < 20:
                    samples_name.append(sample)
                elif not is_name_query and len(samples_random) < 80:
                    # To add some randomness, we don't just take the first 80, 
                    # but taking the first 80 is fine for this large dataset since they are well shuffled usually.
                    samples_random.append(sample)
                    
                if len(samples_name) >= 20 and len(samples_random) >= 80:
                    break
            
            if len(samples_name) >= 20 and len(samples_random) >= 80:
                break
                
    # If we couldn't find enough name samples, fill with random ones
    while len(samples_name) + len(samples_random) < 100:
        samples_random.append(samples_random[-1]) # Just duplicate if we run out (unlikely)
        
    all_samples = samples_name + samples_random
    random.shuffle(all_samples)
    
    print(f"Generating markdown for {len(all_samples)} samples...")
    
    md_content = f"# Korean Heritage VQA - 100 Samples\n\n"
    md_content += f"이 파일에는 총 {len(all_samples)}개의 샘플이 포함되어 있으며, 이름/명칭을 묻는 샘플이 {len(samples_name)}개 포함되어 있습니다.\n\n"
    
    for idx, sample in enumerate(all_samples):
        md_content += f"## Sample {idx + 1}\n\n"
        for key in ['prompt', 'chosen', 'rejected']:
            if key in sample:
                md_content += f"### {key.capitalize()}\n\n"
                part_md, _ = process_messages(sample[key])
                md_content += part_md
        md_content += "---\n\n"
        
    with open(output_md_path, 'w', encoding='utf-8') as fout:
        fout.write(md_content)
        
    print(f"Successfully saved {len(all_samples)} samples to {output_md_path}")

if __name__ == "__main__":
    main()
