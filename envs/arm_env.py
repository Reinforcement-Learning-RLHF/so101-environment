import mujoco
import numpy as np
import collections
from pathlib import Path
from gymnasium import spaces

class ArmEnv:
    def __init__(self, render_width=320, render_height=240, max_steps=400):
        project_root = Path(__file__).parent.parent.resolve()
        xml_path = str(project_root / 'models' / 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 1. Hardware Limits (From your SO-101 XML)
        self.ctrl_limits = np.array([
            [-1.91986, 1.91986], # shoulder_pan
            [-1.74533, 1.74533], # shoulder_lift
            [-1.69,    1.69],    # elbow_flex
            [-1.65806, 1.65806], # wrist_flex
            [-2.74385, 2.84121], # wrist_roll
            [-0.17453, 1.74533], # gripper
        ], dtype=np.float32)

        # 2. Particle Tracking Setup
        self.particle_indices = [
            i for i in range(self.model.nbody)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            and "water_particle" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        ]

        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self.max_steps = max_steps
        self.curr_step = 0

    def get_obs(self):
        obs = collections.OrderedDict()
        # Normalized proprioception (Joint positions)
        # We normalize qpos to [-1, 1] so the Transformer sees consistent scales
        qpos_raw = self.data.qpos[:6].copy()
        low, high = self.ctrl_limits[:, 0], self.ctrl_limits[:, 1]
        qpos_norm = 2.0 * ((qpos_raw - low) / (high - low)) - 1.0
        
        obs['qpos'] = qpos_norm.astype(np.float32)
        
        # Images
        self.renderer.update_scene(self.data, camera="main_observation")
        obs['images/front'] = self.renderer.render()
        self.renderer.update_scene(self.data, camera="wrist_cam")
        obs['images/wrist'] = self.renderer.render()
        
        return obs

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        
        # ✅ RANDOMIZATION: Target Cup
        target_id = self.model.body("target_cup").id
        self.model.body_pos[target_id][:2] = [
            0.5 + np.random.uniform(-0.02, 0.02),
            0.1 + np.random.uniform(-0.02, 0.02)
        ]

        # ✅ RANDOMIZATION: Source Cup & Particles
        source_id = self.model.body("source_cup").id
        s_q_addr = self.model.jnt_qposadr[self.model.body_jntadr[source_id]]
        s_x = 0.5 + np.random.uniform(-0.01, 0.01)
        s_y = -0.1 + np.random.uniform(-0.01, 0.01)
        self.data.qpos[s_q_addr : s_q_addr + 7] = [s_x, s_y, 0.422, 1, 0, 0, 0]

        # Reset Water Particles in a grid above the new source cup position
        particle_count = 0
        grid_side, spacing = 3, 0.009
        for idx in self.particle_indices:
            jnt_addr = self.model.jnt_qposadr[self.model.body_jntadr[idx]]
            ix, iy, iz = (particle_count % 3 - 1), (particle_count // 3 % 3 - 1), (particle_count // 9)
            self.data.qpos[jnt_addr : jnt_addr + 3] = [s_x + ix*spacing, s_y + iy*spacing, 0.435 + iz*spacing]
            particle_count += 1

        # Settle physics
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50): mujoco.mj_step(self.model, self.data)

        self.curr_step = 0
        return self.get_obs()

    def step(self, action):
        self.curr_step += 1
        
        # Denormalize ACT output [-1, 1] to Radian limits
        low, high = self.ctrl_limits[:, 0], self.ctrl_limits[:, 1]
        target_ctrl = low + (np.clip(action, -1, 1) + 1.0) * 0.5 * (high - low)
        self.data.ctrl[:] = target_ctrl

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Reward/Success Logic
        target_pos = self.data.xpos[self.model.body("target_cup").id]
        p_pos = self.data.xpos[self.particle_indices]
        inside = np.all(np.abs(p_pos - target_pos) <= [0.02, 0.02, 0.033], axis=1)
        success_ratio = np.mean(inside)

        obs = self.get_obs()
        done = self.curr_step >= self.max_steps
        return obs, float(success_ratio), done, {"is_success": success_ratio > 0.7}