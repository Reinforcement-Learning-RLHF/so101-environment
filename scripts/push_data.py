"""
Upload Local LeRobot Dataset to Hugging Face Hub

This script loads a local LeRobot dataset and pushes it to the Hugging Face Hub.
It assumes you have already created a dataset locally using LeRobotDataset.create()
and configured your Hugging Face authentication (via `hf auth login`).

run using python scripts/push_data.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load the local dataset (change according to your path and username)
dataset = LeRobotDataset("Ishah8840/so101_pouring", root="data/lerobot/so101_pouring")

# Push to Hugging Face Hub
dataset.push_to_hub()