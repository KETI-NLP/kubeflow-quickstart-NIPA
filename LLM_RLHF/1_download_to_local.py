import os
from datasets import load_dataset

def main():
    print("Downloading Anthropic/hh-rlhf dataset for RLHF/PPO training...")
    # For PPO, we only need the prompts to generate completions against the active policy.
    dataset = load_dataset("Anthropic/hh-rlhf")

    def format_ppo(example):
        chosen = example['chosen']
        rejected = example['rejected']
        
        # Extract common prefix (prompt)
        import os
        common_prefix = os.path.commonprefix([chosen, rejected])
        
        last_assistant_idx = common_prefix.rfind("\n\nAssistant:")
        if last_assistant_idx != -1:
            prompt_end_idx = last_assistant_idx + len("\n\nAssistant:")
            prompt = common_prefix[:prompt_end_idx]
        else:
            prompt = common_prefix

        # PPO dataset needs primarily the "query" (prompt)
        return {
            "query": prompt
        }

    print("Formatting dataset to RLHF/PPO schema (query)...")
    ppo_dataset = dataset.map(format_ppo, remove_columns=dataset['train'].column_names)

    output_dir = "./rlhf_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving formatted dataset to {output_dir}...")
    ppo_dataset.save_to_disk(output_dir)
    print("Download and formatting complete!")

if __name__ == "__main__":
    main()
