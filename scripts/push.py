from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- Configuration ---
# Your Hugging Face repository ID
HF_REPO_ID = "Ishah8840/water_pouring_dataset" 

# The absolute path where the data is stored
LOCAL_DATASET_PATH = "/home/ishan-shah/.cache/huggingface/lerobot/Ishah8840/water_pouring_dataset" 
# ---------------------

def main():
    print(f"Loading local dataset from: {LOCAL_DATASET_PATH}...")
    
    # We must explicitly define BOTH the repo_id and the root directory
    dataset = LeRobotDataset(
        repo_id=HF_REPO_ID,
        root=LOCAL_DATASET_PATH
    )
    
    print(f"Pushing dataset to Hugging Face Hub at: {HF_REPO_ID}...")
    
    # Now it knows the correct repo name to push to
    dataset.push_to_hub()
    
    print("Upload complete! You can view your dataset on Hugging Face.")

if __name__ == "__main__":
    main()