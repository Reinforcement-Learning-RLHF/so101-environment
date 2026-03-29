from huggingface_hub import HfApi

def push_to_hub():
    api = HfApi()
    repo_id = "Ishah8840/so101-water-pour"
    folder_path = "outputs/lerobot_dataset"

    print(f"Creating private repository: {repo_id}...")
    api.create_repo(
        repo_id=repo_id, 
        repo_type="dataset", 
        private=True, 
        exist_ok=True
    )

    print("Uploading dataset (this might take a few minutes)...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print("✅ Successfully pushed to Hugging Face!")

if __name__ == "__main__":
    push_to_hub()