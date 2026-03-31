from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

ds = LeRobotDataset("Ishah8840/so101_pouring", root="data/lerobot/so101_pouring")
# Compare frame 0 and frame 100
f0 = ds[0]["observation.images.wrist"]
f100 = ds[100]["observation.images.wrist"]
print(f"Pixel Difference: {(f0.float() - f100.float()).abs().mean().item()}")