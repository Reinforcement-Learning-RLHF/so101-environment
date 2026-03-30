import time
import numpy as np
import mujoco
from mujoco.viewer import launch_passive
# Assuming your new class name from the previous step
from envs.arm_env import ArmEnv 

# --- Robust IK Solver ---
# (Keep your ik_jacobian function exactly as it is, it works in Radian space)
def ik_jacobian(model, data, target_xyz, site_name="gripperframe", max_iters=100, tol=1e-3, lr=0.5):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    tmp_data = mujoco.MjData(model)
    mujoco.mj_copyData(tmp_data, model, data)
    
    for _ in range(max_iters):
        mujoco.mj_forward(model, tmp_data)
        current_pos = tmp_data.site_xpos[site_id]
        error = target_xyz - current_pos
        if np.linalg.norm(error) < tol: break

        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, tmp_data, jac, None, site_id)
        J = jac[:, :6]
        dq = np.linalg.pinv(J) @ error
        tmp_data.qpos[:6] += lr * dq
        
        # Respect limits
        tmp_data.qpos[:6] = np.clip(tmp_data.qpos[:6], model.jnt_range[:6, 0], model.jnt_range[:6, 1])

    return tmp_data.qpos[:6].copy()

class RandomizedIKPolicy:
    def __init__(self, env):
        self.env = env
        # Pull the hardware limits from the environment class to use for normalization
        self.ctrl_limits = self.env.ctrl_limits 
        self.low = self.ctrl_limits[:, 0]
        self.high = self.ctrl_limits[:, 1]

        # Your original calibrated data (Radians)
        self.perfect_qpos = [
            np.array([0.2197, 0.8423, 0.6051, -1.3397, -1.5400, 1.3000]),
            np.array([0.2243, 1.1886, -0.1385, -1.0597, -1.4800, 1.4853]),
            np.array([0.2264, 1.1224, -0.1192, -1.0015, -1.4799, 0.3881]),
            np.array([-0.1183, 0.4729, 0.2440, -0.5996, -1.4800, 0.3477]),
            np.array([-0.1391, 0.3942, 0.3538, -0.5996, 0.2000, 0.3476])
        ]
        
        self.calibrated_xyz = [
            np.array([0.4617, -0.0801, 0.4398]),
            np.array([0.5228, -0.0955, 0.4399]),
            np.array([0.5265, -0.0971, 0.4492]),
            np.array([0.5275,  0.0315, 0.5209]),
            np.array([0.5300,  0.0400, 0.5300])
        ]

        # Calculate offsets using world positions
        mujoco.mj_forward(self.env.model, self.env.data)
        source_id = self.env.model.body("source_cup").id
        target_id = self.env.model.body("target_cup").id
        self.SOURCE_CUP_HOME = self.env.data.xpos[source_id].copy()
        self.TARGET_CUP_HOME = self.env.data.xpos[target_id].copy()

        self.offsets = [xyz - (self.SOURCE_CUP_HOME if i <= 2 else self.TARGET_CUP_HOME) 
                        for i, xyz in enumerate(self.calibrated_xyz)]

        self.keyframe_steps = [30, 20, 30, 30, 40]
        self.reset()

    def normalize(self, qpos_radians):
        """Converts raw Radians to [-1, 1] for the environment's step()"""
        return 2.0 * ((qpos_radians - self.low) / (self.high - self.low)) - 1.0

    def denormalize(self, qpos_norm):
        """Converts [-1, 1] from obs back to Radians for the IK/Math"""
        return self.low + (qpos_norm + 1.0) * 0.5 * (self.high - self.low)

    def reset(self):
        self.current_frame = 0
        self.step_in_frame = 0
        self.start_qpos_rad = None # Store starting pos in Radians
        
        source_id = self.env.model.body("source_cup").id
        target_id = self.env.model.body("target_cup").id
        curr_source_pos = self.env.data.xpos[source_id]
        curr_target_pos = self.env.data.xpos[target_id]

        self.joint_targets_rad = [] # Targets stored in Radians
        for i, offset in enumerate(self.offsets):
            ref = curr_source_pos if i <= 2 else curr_target_pos
            target_xyz = ref + offset
            
            q_ik = ik_jacobian(self.env.model, self.env.data, target_xyz)
            
            final_q = np.copy(q_ik)
            final_q[4] = self.perfect_qpos[i][4] # Roll
            final_q[5] = self.perfect_qpos[i][5] # Gripper
            
            self.joint_targets_rad.append(final_q)

    def get_action(self, obs):
        if self.current_frame >= len(self.joint_targets_rad):
            # Return last target normalized
            return self.normalize(self.joint_targets_rad[-1])

        target_q_rad = self.joint_targets_rad[self.current_frame]
        total_steps = self.keyframe_steps[self.current_frame]

        if self.step_in_frame == 0:
            # Denormalize the incoming observation so we can do math in Radians
            self.start_qpos_rad = self.denormalize(obs["qpos"][:6])

        # Linear interpolation in Radian space (smoother)
        alpha = (self.step_in_frame + 1) / total_steps
        current_target_rad = (1 - alpha) * self.start_qpos_rad + alpha * target_q_rad

        self.step_in_frame += 1
        if self.step_in_frame >= total_steps:
            self.step_in_frame = 0
            self.current_frame += 1

        # IMPORTANT: Normalize before sending to env.step()
        return self.normalize(current_target_rad)

# --- Verification Loop ---
if __name__ == "__main__":
    env = ArmEnv() 
    policy = RandomizedIKPolicy(env)

    # Note: ArmEnv.reset() returns (obs, info) in newer Gymnasium
    obs = env.reset()
    policy.reset()
    
    with launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # env.step expects normalized [-1, 1] action
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)

            viewer.sync()
            
            if done:
                obs = env.reset()
                policy.reset()