import pandas as pd
import numpy as np

df = pd.read_parquet("data/lerobot/so101_pouring/data/chunk-000/file-000.parquet")
actions = np.stack(df["action"].values)

print(f"Episode frames: {len(df)}")
print(f"Episode duration: {len(df)/50:.1f} seconds")