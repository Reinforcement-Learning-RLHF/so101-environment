import h5py
import numpy as np
import glob

files = glob.glob("act_dataset_*/episode_*.hdf5")

all_states = []
all_actions = []

for file in files:
    with h5py.File(file, "r") as f:
        qpos = f["observations"]["qpos"][:]
        qvel = f["observations"]["qvel"][:]
        action = f["action"][:]

        state = np.concatenate([qpos, qvel], axis=-1)

        all_states.append(state)
        all_actions.append(action)

states = np.concatenate(all_states)
actions = np.concatenate(all_actions)

print(states.shape, actions.shape)