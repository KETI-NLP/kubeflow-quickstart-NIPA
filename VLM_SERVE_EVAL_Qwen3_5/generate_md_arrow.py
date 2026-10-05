import pyarrow as pa
import base64
import json

def image_to_base64(img_bytes):
    try:
        img_str = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        return f"Error converting image: {e}"

def process_content(content_list):
    md = ""
    # if it's a numpy array, convert to list
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
                md += f"{item['text']}\n\n"
    return md

def process_messages(messages):
    md = ""
    if hasattr(messages, 'tolist'):
        messages = messages.tolist()
        
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown').capitalize()
            md += f"#### {role}:\n\n"
            if 'content' in msg:
                md += process_content(msg['content'])
    return md

def read_sample_from_arrow():
    arrow_path = "/workspace/2026_llm_data_generation/llm_training_ready/data_korean_heritage_vqa_hf/train/data-00000-of-00486.arrow"
    
    with pa.OSFile(arrow_path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        batch = next(reader)
        
    df = batch.to_pandas()
    sample = df.iloc[0].to_dict()
    
    output_md_path = "/workspace/kubeflow-training-snippets/VLM_SERVE_EVAL_Qwen3_5/korean_heritage_vqa_sample.md"
    
    md_content = f"# Dataset Sample\n\n"
    
    for key in ['prompt', 'chosen', 'rejected']:
        if key in sample:
            md_content += f"## {key.capitalize()}\n\n"
            md_content += process_messages(sample[key])
            md_content += "---\n\n"
            
    with open(output_md_path, 'w', encoding='utf-8') as fout:
        fout.write(md_content)
    print(f"Successfully wrote sample to {output_md_path}")

if __name__ == "__main__":
    read_sample_from_arrow()
