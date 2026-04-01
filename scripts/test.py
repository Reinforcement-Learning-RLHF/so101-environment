from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load the local dataset
dataset = LeRobotDataset("Ishah8840/so101_pouring", root="data/lerobot/so101_pouring")

# Push to Hugging Face Hub
dataset.push_to_hub()