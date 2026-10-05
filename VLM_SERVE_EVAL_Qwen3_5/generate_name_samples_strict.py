import pyarrow as pa
import base64
import re
from io import BytesIO
from PIL import Image

def image_to_base64(img_bytes):
    try:
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
    output_md_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/korean_heritage_vqa_name_samples.md"
    
    samples_name = []
    
    # EXACT whitelist patterns for asking the identity/name of the main heritage
    include_patterns = [
        re.compile(r"이\s*(문화재|유물|불상|탑|건축물|작품)의\s*(이름|명칭)"),
        re.compile(r"(사진|이미지)\s*속\s*(문화재|유물|불상|탑|건축물|작품)의\s*(이름|명칭)"),
        re.compile(r"(문화재|유물|불상|탑|건축물|작품)의\s*정확한\s*(이름|명칭)"),
        re.compile(r"어떤\s*(문화재|유물|불상|탑)인가요")
    ]
    
    # Additional strict blacklist to completely wipe out material/part names
    exclude_pattern = re.compile(r"(부분|재료|양식|재질|나무|종이|지붕|석재|기법|바위|섬|산|글씨|한자|문구|사찰|종류|옷|무늬|기둥|색상|색깔|명칭이 포함)")
    
    import glob
    arrow_files = sorted(glob.glob("/workspace/2026_llm_data_generation/llm_training_ready/data_korean_heritage_vqa_hf/train/data-*-of-00486.arrow"))
    
    for file_path in arrow_files:
        print(f"Processing {file_path}...")
        with pa.OSFile(file_path, 'rb') as f:
            reader = pa.ipc.open_stream(f)
            
            for batch in reader:
                df = batch.to_pandas()
                for i in range(len(df)):
                    sample = df.iloc[i].to_dict()
                    
                    _, prompt_text = process_messages(sample.get('prompt', []))
                    
                    is_match = any(p.search(prompt_text) for p in include_patterns)
                    is_excluded = exclude_pattern.search(prompt_text)
                    
                    if is_match and not is_excluded:
                        samples_name.append(sample)
                        
                    if len(samples_name) >= 100:
                        break
                
                if len(samples_name) >= 100:
                    break
        if len(samples_name) >= 100:
            break
                
    print(f"Generating markdown for {len(samples_name)} strict name samples...")
    
    md_content = f"# Korean Heritage VQA - Strict Name/Title Samples\n\n"
    md_content += f"이 파일에는 재질이나 양식, 부분 명칭이 아닌, 문화재/유물 자체의 이름이나 명칭을 묻는 엄격하게 필터링된 샘플 {len(samples_name)}개가 포함되어 있습니다.\n\n"
    
    for idx, sample in enumerate(samples_name):
        md_content += f"## Sample {idx + 1}\n\n"
        for key in ['prompt', 'chosen', 'rejected']:
            if key in sample:
                md_content += f"### {key.capitalize()}\n\n"
                part_md, _ = process_messages(sample[key])
                md_content += part_md
        md_content += "---\n\n"
        
    with open(output_md_path, 'w', encoding='utf-8') as fout:
        fout.write(md_content)
        
    print(f"Successfully saved {len(samples_name)} samples to {output_md_path}")

if __name__ == "__main__":
    main()
