import pyarrow as pa
import re

def main():
    arrow_path = "/workspace/2026_llm_data_generation/llm_training_ready/data_korean_heritage_vqa_hf/train/data-00000-of-00486.arrow"
    
    print(f"Reading from {arrow_path}...")
    
    pattern = re.compile(r"(문화재|유물|유적|작품|건축물|사진 속).*?(이름|명칭).*?(무엇|어떻게|식별|알려)")
    samples_found = 0
    
    with pa.OSFile(arrow_path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        
        for batch in reader:
            df = batch.to_pandas()
            for i in range(len(df)):
                sample = df.iloc[i].to_dict()
                
                messages = sample.get('prompt', [])
                if hasattr(messages, 'tolist'):
                    messages = messages.tolist()
                
                prompt_text = ""
                for msg in messages:
                    if isinstance(msg, dict) and 'content' in msg:
                        content = msg['content']
                        if hasattr(content, 'tolist'):
                            content = content.tolist()
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                prompt_text += item.get('text', '')
                
                # To exclude material/tree/style names
                exclude_pattern = re.compile(r"(양식|재질|나무|종이|지붕|석재).*?(명칭|이름)")
                
                if pattern.search(prompt_text) and not exclude_pattern.search(prompt_text):
                    print(f"- {prompt_text}")
                    samples_found += 1
                
                if samples_found >= 20:
                    return

if __name__ == "__main__":
    main()
