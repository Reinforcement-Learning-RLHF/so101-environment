import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


class ArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_width=320, render_height=240, max_steps=400):
        super().__init__()

        project_root = Path(__file__).parent.parent.resolve()
        xml_path = str(project_root / 'models' / 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.ctrl_limits = np.array([
            [-1.91986, 1.91986],  # shoulder_pan
            [-1.74533, 1.74533],  # shoulder_lift
            [-1.69,    1.69],     # elbow_flex
            [-1.65806, 1.65806],  # wrist_flex
            [-2.74385, 2.84121],  # wrist_roll
            [-0.17453, 1.74533],  # gripper
        ], dtype=np.float32)

        self.particle_indices = [
            i for i in range(self.model.nbody)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            and "water_particle" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        ]

        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self.max_steps = max_steps
        self.render_width = render_width
        self.render_height = render_height
        self.curr_step = 0

        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict({
                "front": spaces.Box(0, 255, (render_height, render_width, 3), dtype=np.uint8),
                "wrist": spaces.Box(0, 255, (render_height, render_width, 3), dtype=np.uint8),
            }),
            # Raw joint positions in radians
            "agent_pos": spaces.Box(
                low=self.ctrl_limits[:, 0],
                high=self.ctrl_limits[:, 1],
                dtype=np.float32,
            ),
        })

        # Raw radian targets — LeRobot will normalize for the policy
        self.action_space = spaces.Box(
            low=self.ctrl_limits[:, 0],
            high=self.ctrl_limits[:, 1],
            dtype=np.float32,
        )

    def get_obs(self):
        self.renderer.update_scene(self.data, camera="main_observation")
        front = self.renderer.render().copy()

        self.renderer.update_scene(self.data, camera="wrist_cam")
        wrist = self.renderer.render().copy()

        return {
            "pixels": {
                "front": front,
                "wrist": wrist,
            },
            # Raw radians — no normalization here
            "agent_pos": self.data.qpos[:6].copy().astype(np.float32),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize target cup position
        target_id = self.model.body("target_cup").id
        self.model.body_pos[target_id][:2] = [
            0.5 + np.random.uniform(-0.02, 0.02),
            0.1 + np.random.uniform(-0.02, 0.02)
        ]

        # Randomize source cup position
        source_id = self.model.body("source_cup").id
        s_q_addr = self.model.jnt_qposadr[self.model.body_jntadr[source_id]]
        s_x = 0.5 + np.random.uniform(-0.01, 0.01)
        s_y = -0.1 + np.random.uniform(-0.01, 0.01)
        self.data.qpos[s_q_addr:s_q_addr + 7] = [s_x, s_y, 0.422, 1, 0, 0, 0]

        # Reset water particles in a grid above source cup
        particle_count = 0
        spacing = 0.009
        for idx in self.particle_indices:
            jnt_addr = self.model.jnt_qposadr[self.model.body_jntadr[idx]]
            ix = particle_count % 3 - 1
            iy = particle_count // 3 % 3 - 1
            iz = particle_count // 9
            self.data.qpos[jnt_addr:jnt_addr + 3] = [
                s_x + ix * spacing,
                s_y + iy * spacing,
                0.435 + iz * spacing,
            ]
            particle_count += 1

        # Settle physics
        mujoco.mj_forward(self.model, self.data)
        
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self.curr_step = 0
        return self.get_obs(), {"is_success": False}

    def step(self, action):
        self.curr_step += 1

        self.data.ctrl[:] = np.clip(
            action,
            self.ctrl_limits[:, 0],
            self.ctrl_limits[:, 1],
        )

        # policy 1
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # mujoco.mj_step(self.model, self.data)

        # Success: fraction of water particles inside target cup
        target_pos = self.data.xpos[self.model.body("target_cup").id]
        p_pos = self.data.xpos[self.particle_indices]
        inside = np.all(np.abs(p_pos - target_pos) <= [0.02, 0.02, 0.033], axis=1)
        success_ratio = float(np.mean(inside))

        terminated = False
        success = success_ratio > 0.7
        truncated = self.curr_step >= self.max_steps

        return self.get_obs(), success_ratio, terminated, truncated, {"is_success": success}

    def close(self):
        self.renderer.close()


def make_env(n_envs: int = 1, use_async_envs: bool = False):
    def _make():
        return ArmEnv()

    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return env_cls([_make for _ in range(n_envs)])