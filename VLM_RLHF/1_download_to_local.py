import os
from datasets import load_dataset

def main():
    dataset_name = "openbmb/RLHF-V-Dataset"
    save_path = "./vlm_rlhf_dataset"
    output_dir = save_path
    print(f"Downloading {dataset_name} dataset for RLHF/PPO training...")
    dataset = load_dataset(dataset_name)

    import json

    def format_ppo(example):
        """
        PPO Trainer expects just a 'prompt' string or message list.
        For RLHF-V, we extract the question and format it as a prompt.
        """
        try:
            parsed = json.loads(example["text"])
            prompt_str = parsed.get("question", "")
        except:
            prompt_str = ""
            
        return {
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_str}]}],
            "images": [example["image"]] if example.get("image") is not None else []
        }

    print("Formatting dataset to RLHF/PPO schema (prompt, images)...")
    ppo_dataset = dataset.map(format_ppo, remove_columns=dataset['train'].column_names)

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving formatted dataset to {output_dir}...")
    ppo_dataset.save_to_disk(output_dir)
    print("Download and formatting complete!")

if __name__ == "__main__":
    main()
