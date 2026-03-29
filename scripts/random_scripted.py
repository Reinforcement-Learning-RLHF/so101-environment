import time
import numpy as np
import mujoco
from mujoco.viewer import launch_passive
from envs.arm_env import ArmEnv

# --- Robust IK Solver ---
def ik_jacobian(model, data, target_xyz, site_name="gripperframe", max_iters=100, tol=1e-3, lr=0.5):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found! Ensure it is in your XML.")

    # Use a copy of data to avoid messing up the actual simulation state during iteration
    tmp_data = mujoco.MjData(model)
    mujoco.mj_copyData(tmp_data, model, data)
    
    for _ in range(max_iters):
        mujoco.mj_forward(model, tmp_data)
        current_pos = tmp_data.site_xpos[site_id]
        error = target_xyz - current_pos
        
        if np.linalg.norm(error) < tol:
            break

        # Get the Jacobian for the site
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, tmp_data, jac, None, site_id)
        
        # Focus on the first 6 arm joints
        J = jac[:, :6]
        
        # Damped Least Squares update
        dq = np.linalg.pinv(J) @ error
        tmp_data.qpos[:6] += lr * dq
        
        # Respect joint limits from XML
        tmp_data.qpos[:6] = np.clip(
            tmp_data.qpos[:6], 
            model.jnt_range[:6, 0], 
            model.jnt_range[:6, 1]
        )

    return tmp_data.qpos[:6].copy()

# --- Policy Class ---
class RandomizedIKPolicy:
    def __init__(self, env):
        self.env = env
        
        # 1. Your Calibrated QPOS (Keep these as the "Rotation/Gripper" reference)
        self.perfect_qpos = [
            np.array([0.2197, 0.8423, 0.6051, -1.3397, -1.5400, 1.3000]),
            np.array([0.2243, 1.1886, -0.1385, -1.0597, -1.4800, 1.4853]),
            np.array([0.2264, 1.1224, -0.1192, -1.0015, -1.4799, 0.3881]),
            np.array([-0.1183, 0.4729, 0.2440, -0.5996, -1.4800, 0.3477]),
            np.array([-0.1391, 0.3942, 0.3538, -0.5996, 0.2000, 0.3476])
        ]
        
        # 2. Your Calibrated XYZ (The ones you got from your test run)
        self.calibrated_xyz = [
            np.array([0.4617, -0.0801, 0.4398]),
            np.array([0.5228, -0.0955, 0.4399]),
            np.array([0.5265, -0.0971, 0.4492]),
            np.array([0.5275,  0.0315, 0.5209]),
            np.array([0.5300,  0.0400, 0.5300])
        ]
        
        # --- THE FIX: DYNAMIC HOME POSITIONS ---
        # We use the body names defined in your CupEnv ('source_cup' and 'target_cup')
        source_id = self.env.model.body("source_cup").id
        target_id = self.env.model.body("target_cup").id

        # Use model.body_pos to get the "Reference" position from the XML
        # Note: If they move during reset, data.xpos is better, but for 
        # offset calculation, we use the state when you did the calibration.
        self.SOURCE_CUP_HOME = self.env.model.body_pos[source_id].copy()
        self.TARGET_CUP_HOME = self.env.model.body_pos[target_id].copy()
        
        # In your CupEnv, source_cup uses a joint, so its 'home' might be in qpos
        # Let's pull the actual world position after a forward pass to be safe
        mujoco.mj_forward(self.env.model, self.env.data)
        self.SOURCE_CUP_HOME = self.env.data.xpos[source_id].copy()
        self.TARGET_CUP_HOME = self.env.data.xpos[target_id].copy()
        # ---------------------------------------

        # Calculate Offsets relative to the actual env positions
        self.offsets = [xyz - (self.SOURCE_CUP_HOME if i <= 2 else self.TARGET_CUP_HOME) 
                        for i, xyz in enumerate(self.calibrated_xyz)]

        self.keyframe_steps = [40, 60, 40, 80, 60]
        self.reset()

    def reset(self):
        self.current_frame = 0
        self.step_in_frame = 0
        self.start_qpos = None
        
        source_id = self.env.model.body("source_cup").id
        target_id = self.env.model.body("target_cup").id
        curr_source_pos = self.env.data.xpos[source_id]
        curr_target_pos = self.env.data.xpos[target_id]

        self.joint_targets = []
        for i, offset in enumerate(self.offsets):
            ref = curr_source_pos if i <= 2 else curr_target_pos
            target_xyz = ref + offset
            
            # Solve IK for the arm position (Joints 0-3)
            q_ik = ik_jacobian(self.env.model, self.env.data, target_xyz)
            
            # Combine IK position with your TELEOP Rotation and Gripper
            # We take index 4 (Roll) and index 5 (Gripper) from your teleop data
            final_q = np.copy(q_ik)
            final_q[4] = self.perfect_qpos[i][4] # Wrist Roll
            final_q[5] = self.perfect_qpos[i][5] # Gripper
            
            self.joint_targets.append(final_q)

    def get_action(self, obs):
        if self.current_frame >= len(self.joint_targets):
            return self.joint_targets[-1]

        target_qpos = self.joint_targets[self.current_frame]
        total_steps = self.keyframe_steps[self.current_frame]

        if self.step_in_frame == 0:
            self.start_qpos = obs["qpos"][:6]

        alpha = (self.step_in_frame + 1) / total_steps
        action = (1 - alpha) * self.start_qpos + alpha * target_qpos

        self.step_in_frame += 1
        if self.step_in_frame >= total_steps:
            self.step_in_frame = 0
            self.current_frame += 1

        return action

# --- Verification Loop ---
if __name__ == "__main__":
    # Initialize Environment
    env = ArmEnv() 
    policy = RandomizedIKPolicy(env)

    # Reset both to start fresh
    obs = env.reset()
    policy.reset()
    
    print("Launching Passive Viewer... Press ESC in viewer to quit.")
    
    with launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # 1. Get smooth joint command from policy
            action = policy.get_action(obs)
            
            # 2. Step the simulation
            obs, reward, done, info = env.step(action)

            # 3. Refresh viewer
            viewer.sync()
            
            # Maintain roughly real-time speed (0.01s matches ~100Hz)
            time.sleep(0.01)
            
            # Auto-reset loop for visual testing
            if done:
                print("Episode Finished. Resetting...")
                obs = env.reset()
                policy.reset()