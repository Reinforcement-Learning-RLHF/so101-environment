import mujoco
import numpy as np
from pathlib import Path
from gymnasium import spaces

class CupEnv:
    def __init__(self, render_width=320, render_height=240, max_steps=400, render_images=False):
        project_root = Path(__file__).parent.parent.resolve()
        xml_path = str(project_root / 'models' / 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Dimensions
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.action_dim = len(self.joint_names)
        self.obs_dim = len(self.joint_names)

        self.joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        }

        self.n_substeps = 10

        # Renderer
        self.render_images = render_images
        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self.cam_names = ["main_observation", "wrist_cam"]

        # Gym-style spaces
        self.action_space = spaces.Box(low=-3.14, high=3.14, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Episode tracking
        self.max_steps = max_steps
        self.curr_step = 0
        self.episode_id = None

        self.particle_indices = [
            i for i in range(self.model.nbody)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            and "water_particle" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        ]

    def get_image(self, cam_name):
        self.renderer.update_scene(self.data, camera=cam_name)
        img = self.renderer.render()
        img = np.transpose(img, (2, 0, 1)) / 255.0  
        return img.astype(np.float32)

    def get_obs(self):
        qpos = self.data.qpos[:6].copy()
        qvel = self.data.qvel[:6].copy()
        
        obs = {
            "qpos": qpos,
            "qvel": qvel,
            "episode_id": self.episode_id,
            "step": self.curr_step
        }

        if self.render_images:
            main_img = self.get_image("main_observation")
            wrist_img = self.get_image("wrist_cam")
            obs["image_main"] = main_img
            obs["image_wrist"] = wrist_img

        return obs

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Fixed episode ID for reproducibility
        self.episode_id = 0 

        # 1. FIXED target cup position (removed uniform noise)
        target_id = self.model.body("target_cup").id
        self.model.body_pos[target_id][:2] = [0.5, 0.1]

        # 2. FIXED source cup position (removed uniform noise)
        source_id = self.model.body("source_cup").id
        s_q_addr = self.model.jnt_qposadr[self.model.body_jntadr[source_id]]
        s_x = 0.5 
        s_y = -0.1
        self.data.qpos[s_q_addr : s_q_addr + 7] = [s_x, s_y, 0.422, 1, 0, 0, 0]

        # 3. Particle Grid (Logic remains, but inputs s_x and s_y are now fixed)
        particle_count = 0
        grid_side = 3  
        spacing = 0.009 
        
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and "water_particle" in name:
                # Find the starting address of this body's joint in qpos
                jnt_addr = self.model.jnt_qposadr[self.model.body_jntadr[i]]
                
                ix = (particle_count % grid_side) - 1
                iy = ((particle_count // grid_side) % grid_side) - 1
                iz = (particle_count // (grid_side * grid_side))
                
                # Update 7 slots: [x, y, z, qw, qx, qy, qz]
                # If you only update 3, the orientation (quaternion) stays 0,0,0,0 
                # which is an invalid rotation and causes physics to "pop" or vanish.
                self.data.qpos[jnt_addr : jnt_addr + 7] = [
                    s_x + (ix * spacing), 
                    s_y + (iy * spacing), 
                    0.435 + (iz * spacing),
                    1.0, 0.0, 0.0, 0.0  # Unit quaternion (identity rotation)
                ]
                
                # Reset velocities (6 slots for freejoint: 3 trans, 3 rot)
                v_addr = self.model.jnt_dofadr[self.model.body_jntadr[i]]
                self.data.qvel[v_addr : v_addr + 6] = 0.0
                particle_count += 1

        # 4. REMOVED Domain Randomization
        table_id = self.model.geom("table_top").id
        self.model.geom_friction[table_id][0] = 1.0 # Fixed friction
        self.model.body_mass[source_id] = 0.055     # Fixed mass
        
        # Fixed light and camera
        self.model.light_diffuse[0] = [0.65, 0.65, 0.65] 
        cam_id = self.model.camera("main_observation").id
        self.model.cam_pos[cam_id] = np.array([1.2, 0.0, 1.3])

        # 5. Settle Period (Kept for numerical stability, but deterministic)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
            
        self.curr_step = 0 
        return self.get_obs()

    def step(self, action):
        self.curr_step += 1

        # REMOVED action noise
        self.data.ctrl[:] = np.clip(action, self.action_space.low, self.action_space.high)

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
        done = success_ratio > 0.9 or self.curr_step >= self.max_steps
        reward = success_ratio 

        return obs, reward, done, {"is_success": success_ratio > 0.9}

    def close(self):
        pass