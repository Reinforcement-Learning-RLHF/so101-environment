import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def convert():
    # 1. Setup paths - change these to match your folder names
    raw_dir = Path("dataset") 
    repo_id = "ishan-shah/so101-water-pour"
    local_root = Path("outputs/lerobot_dataset") 
    
    # 2. Define the dataset structure
    # Based on your error log, the images are (240, 320, 3)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=local_root,
        fps=60, 
        robot_type="so101",
        features={
            "observation.state": {"dtype": "float32", "shape": (6,)},
            "observation.images.main": {"dtype": "video", "shape": (240, 320, 3), "names": ["color"]},
            "observation.images.wrist": {"dtype": "video", "shape": (240, 320, 3), "names": ["color"]},
            "action": {"dtype": "float32", "shape": (6,)},
        }
    )

    # 3. Loop through your 59 files
    files = sorted([f for f in raw_dir.glob("*.hdf5")])
    
    for file_path in tqdm(files, desc="Converting Episodes"):
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations/qpos'][:].astype(np.float32)
            action = f['action'][:].astype(np.float32)
            
            # Load raw images (currently [C, H, W])
            img_main_raw = f['observations/images/main_observation'][:]
            img_wrist_raw = f['observations/images/wrist_cam'][:]

            # Transpose from (Batch, C, H, W) to (Batch, H, W, C)
            # This moves axis 1 to the last position
            img_main = np.transpose(img_main_raw, (0, 2, 3, 1))
            img_wrist = np.transpose(img_wrist_raw, (0, 2, 3, 1))

            for i in range(len(action)):
                dataset.add_frame({
                    "observation.state": qpos[i],
                    "observation.images.main": img_main[i],
                    "observation.images.wrist": img_wrist[i],
                    "action": action[i],
                    "task": "pour water into the cup", 
                })
            dataset.save_episode()

    # 4. Finalize writes the stats.json and meta folders
    dataset.finalize()
    
    print(f"✅ Done! Dataset saved locally to: {local_root.absolute()}")

if __name__ == "__main__":
    convert()