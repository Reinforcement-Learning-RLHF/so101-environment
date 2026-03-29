import numpy as np
import mujoco
from envs.cup_env import CupEnv

# 1. Setup Environment
env = CupEnv(render_images=False) 
model = env.model
data = env.data

# 2. Identify the target site (from your XML)
site_name = "gripperframe"
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

if site_id == -1:
    raise ValueError(f"Site '{site_name}' not found! Check your XML for the correct name.")

# 3. Your "Perfect" Keyframes from Teleop
perfect_keyframes = [
    {"target": np.array([0.2197, 0.8423, 0.6051, -1.3397, -1.5400, 1.3000])},
    {"target": np.array([0.2243, 1.1886, -0.1385, -1.0597, -1.4800, 1.4853])},
    {"target": np.array([0.2264, 1.1224, -0.1192, -1.0015, -1.4799, 0.3881])},
    {"target": np.array([-0.1183, 0.4729, 0.2440, -0.5996, -1.4800, 0.3477])},
    {"target": np.array([-0.1391, 0.3942, 0.3538, -0.5996, 0.2000, 0.3476])}
]

print("--- SO-101 Cartesian Calibration ---")
for i, frame in enumerate(perfect_keyframes):
    # Set the robot to the specific joint configuration
    data.qpos[:6] = frame["target"]
    
    # Forward kinematics: Calculate where the gripper is in 3D space
    mujoco.mj_forward(model, data)
    
    # Get the world coordinates of the 'gripperframe' site
    ee_pos = data.site_xpos[site_id].copy()
    
    print(f"Frame {i}:")
    print(f"  Joints: {frame['target']}")
    print(f"  XYZ:    {ee_pos}\n")