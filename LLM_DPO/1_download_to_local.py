import os
from datasets import load_dataset

def main():
    print("Downloading Anthropic/hh-rlhf dataset for DPO training...")
    # Load the helpfulness/harmlessness preference dataset
    dataset = load_dataset("Anthropic/hh-rlhf")

    def format_dpo(example):
        """
        The Anthropic dataset contains 'chosen' and 'rejected' strings which include the prompt.
        We need to extract the common prompt and split the responses.
        For simplicity, we handle the common extraction logic.
        """
        chosen = example['chosen']
        rejected = example['rejected']
        
        # In Anthropic hh-rlhf, both chosen and rejected start with the exact same dialogue history (prompt)
        # We find the longest common prefix which represents the prompt
        import os
        common_prefix = os.path.commonprefix([chosen, rejected])
        
        # Typically the last Assistant turn separates the prompt from the response
        last_assistant_idx = common_prefix.rfind("\n\nAssistant:")
        if last_assistant_idx != -1:
            prompt_end_idx = last_assistant_idx + len("\n\nAssistant:")
            prompt = common_prefix[:prompt_end_idx]
            chosen_response = chosen[prompt_end_idx:]
            rejected_response = rejected[prompt_end_idx:]
        else:
            prompt = common_prefix
            chosen_response = chosen[len(common_prefix):]
            rejected_response = rejected[len(common_prefix):]

        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }

    print("Formatting dataset to DPO schema (prompt, chosen, rejected)...")
    dpo_dataset = dataset.map(format_dpo, remove_columns=dataset['train'].column_names)

    output_dir = "./dpo_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving formatted dataset to {output_dir}...")
    dpo_dataset.save_to_disk(output_dir)
    print("Download and formatting complete!")

if __name__ == "__main__":
    main()
