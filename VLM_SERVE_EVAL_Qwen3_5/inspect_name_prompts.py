import pyarrow as pa
import json

def main():
    arrow_path = "/workspace/2026_llm_data_generation/llm_training_ready/data_korean_heritage_vqa_hf/train/data-00000-of-00486.arrow"
    
    print(f"Reading from {arrow_path}...")
    
    samples_found = 0
    
    with pa.OSFile(arrow_path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        
        for batch in reader:
            df = batch.to_pandas()
            for i in range(len(df)):
                sample = df.iloc[i].to_dict()
                
                # Check prompt text
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
                
                if "이름" in prompt_text or "명칭" in prompt_text:
                    print(f"- {prompt_text}")
                    samples_found += 1
                
                if samples_found >= 20:
                    return

if __name__ == "__main__":
    main()
