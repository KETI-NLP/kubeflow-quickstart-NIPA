import os
from datasets import load_dataset

def main():
    dataset_name = "yahma/alpaca-cleaned"
    save_path = "./alpaca_dataset"
    
    print(f"Downloading '{dataset_name}' from HuggingFace...")
    # SFT용 데이터셋 다운로드
    dataset = load_dataset(dataset_name)
    
    # 로컬 디스크에 저장 (추후 이 폴더를 MLXP 환경으로 업로드)
    dataset.save_to_disk(save_path)
    
    print(f"Dataset successfully saved locally to: {os.path.abspath(save_path)}")
    print("Next step: Upload this folder to MLXP DataManager.")

if __name__ == "__main__":
    main()
