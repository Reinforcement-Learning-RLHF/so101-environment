from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

ds = LeRobotDataset("Ishah8840/water_pouring_dataset")

# Collect state stats per joint across first 1000 frames
states = torch.stack([ds[i]["observation.state"] for i in range(1000)])
print("per-joint mean:", states.mean(0))
print("per-joint std: ", states.std(0))
print("per-joint min: ", states.min(0).values)
print("per-joint max: ", states.max(0).values)