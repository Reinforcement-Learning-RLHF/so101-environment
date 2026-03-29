import mujoco
import numpy as np
from pathlib import Path
from gymnasium import spaces

class ArmEnv:
    def __init__(self, render_width=320, render_height=240, max_steps=400, render_images=False):
        project_root = Path(__file__).parent.parent.resolve()
        xml_path = str(project_root / 'models' / 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Dimensions
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.action_dim = len(self.joint_names)
        self.joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        }
        self.n_substeps = 10
        
        # Renderer
        self.render_images = render_images
        self.render_width = render_width
        self.render_height = render_height
        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self.cam_names = ["main_observation", "wrist_cam"]
        
        # 1. FIXED: Gym-style spaces
        self.action_space = spaces.Box(low=-3.14, high=3.14, shape=(self.action_dim,), dtype=np.float32)
        
        # Define Observation Space as a Dict to match get_obs()
        obs_spaces = {
            "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "qvel": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "episode_id": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int32), 
            "step": spaces.Box(low=0, high=max_steps + 1, shape=(), dtype=np.int32)
        }
        
        if self.render_images:
            # Standard Gym image format: (H, W, C), uint8, [0, 255]
            obs_spaces["image_main"] = spaces.Box(low=0, high=255, shape=(render_height, render_width, 3), dtype=np.uint8)
            obs_spaces["image_wrist"] = spaces.Box(low=0, high=255, shape=(render_height, render_width, 3), dtype=np.uint8)
            
        self.observation_space = spaces.Dict(obs_spaces)
        
        # Episode tracking
        self.max_steps = max_steps
        self.curr_step = 0
        self.episode_id = None
        self.particle_indices = [
            i for i in range(self.model.nbody)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            and "water_particle" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        ]

    # 2. FIXED: Keep as standard (H, W, C) uint8 for LeRobot dataloaders
    def get_image(self, cam_name):
        self.renderer.update_scene(self.data, camera=cam_name)
        img = self.renderer.render()
        return img 

    def get_obs(self):
        qpos = self.data.qpos[:6].copy().astype(np.float32)
        qvel = self.data.qvel[:6].copy().astype(np.float32)
        
        obs = {
            "qpos": qpos,
            "qvel": qvel,
            "episode_id": np.array(self.episode_id, dtype=np.int32),
            "step": np.array(self.curr_step, dtype=np.int32)
        }
        if self.render_images:
            obs["image_main"] = self.get_image("main_observation")
            obs["image_wrist"] = self.get_image("wrist_cam")
        return obs

    # 3. FIXED: Reset takes seed/options and returns obs, info
    def reset(self, seed=None, options=None): 
        if seed is not None:
            np.random.seed(seed)
            
        mujoco.mj_resetData(self.model, self.data)
        self.episode_id = np.random.randint(0, int(1e6))
        self.data.qpos[self.joint_ids["wrist_roll"]] = -np.pi/2
        
        # Randomize target cup
        target_id = self.model.body("target_cup").id
        self.model.body_pos[target_id][:2] = [0.5 + np.random.uniform(-0.02, 0.02), 
                                            0.1 + np.random.uniform(-0.02, 0.02)]
        
        # Randomize source cup
        source_id = self.model.body("source_cup").id
        s_q_addr = self.model.jnt_qposadr[self.model.body_jntadr[source_id]]
        s_x = 0.5 + np.random.uniform(-0.02, 0.02)
        s_y = -0.1 + np.random.uniform(-0.02, 0.02)
        self.data.qpos[s_q_addr : s_q_addr + 7] = [s_x, s_y, 0.422, 1, 0, 0, 0]
        
        # Move particles into a 3D grid
        particle_count = 0
        grid_side = 3 
        spacing = 0.009 
        
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and "water_particle" in name:
                jnt_addr = self.model.jnt_qposadr[self.model.body_jntadr[i]]
                
                # Calculate grid offsets
                ix = (particle_count % grid_side) - 1
                iy = ((particle_count // grid_side) % grid_side) - 1
                iz = (particle_count // (grid_side * grid_side))
                
                self.data.qpos[jnt_addr : jnt_addr + 3] = [
                    s_x + (ix * spacing), 
                    s_y + (iy * spacing), 
                    0.435 + (iz * spacing)
                ]
                
                # Reset velocities
                v_addr = self.model.jnt_dofadr[self.model.body_jntadr[i]]
                self.data.qvel[v_addr : v_addr + 6] = 0.0
                particle_count += 1
                
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
            
        self.curr_step = 0 
        
        return self.get_obs(), {}

    def step(self, action):
        self.curr_step += 1
        noise_std = 0.0002 
    
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_std, size=action.shape)
        noisy_action = action + noise
        
        # Clip action
        self.data.ctrl[:] = np.clip(noisy_action, self.action_space.low, self.action_space.high)
        
        # Advance physics
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            
        obs = self.get_obs()
        
        # Check particles inside target cup
        target_pos = self.data.xpos[self.model.body("target_cup").id]
        cup_halfsize = np.array([0.02, 0.02, 0.033])
        p_pos = self.data.xpos[self.particle_indices]
        inside = np.all(np.abs(p_pos - target_pos) <= cup_halfsize, axis=1)
        
        success_ratio = np.mean(inside)
        
        # 4. FIXED: 5-tuple Gym API return
        terminated = bool(success_ratio > 0.9)
        truncated = bool(self.curr_step >= self.max_steps)
        reward = float(success_ratio)
        
        info = {"is_success": terminated}
        
        return obs, reward, terminated, truncated, info

    def close(self):
        pass