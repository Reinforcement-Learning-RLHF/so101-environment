# Command used to run this script: sudo PYTHONPATH=. $(which python) scripts/teleop.py

import numpy as np
import mujoco
import mujoco.viewer
import keyboard
from envs.arm_env import ArmEnv

def main():
    env = ArmEnv(render_images=False)
    env.reset()

    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')

    # Actuator limits from so101.xml
    lower_limits = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
    upper_limits = np.array([ 1.91986,  1.74533,  1.69,  1.65806,  2.84121,  1.74533])

    current_action = np.copy(env.data.qpos[:6])

    # Tuning parameters
    ik_step_size = 0.015  
    gripper_step = 0.15   
    damping = 0.05        # Damping factor for DLS IK to prevent explosions at full reach
    max_lead = 0.15       # The "leash" - max radians the target can get ahead of physical joints

    print("=== SO101 Robotic Arm Teleop (Damped) ===")
    print(" W/S   : Move Forward/Backward (X)")
    print(" A/D   : Move Left/Right (Y)")
    print(" Q/E   : Move Up/Down (Z)")
    print(" SPACE : Close Gripper")
    print(" C     : Open Gripper")
    print(" ESC   : Quit")
    print("=========================================")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            if keyboard.is_pressed('esc'):
                break

            dx = np.zeros(3)
            if keyboard.is_pressed('w'): dx[0] += ik_step_size
            if keyboard.is_pressed('s'): dx[0] -= ik_step_size
            if keyboard.is_pressed('a'): dx[1] += ik_step_size
            if keyboard.is_pressed('d'): dx[1] -= ik_step_size
            if keyboard.is_pressed('q'): dx[2] += ik_step_size
            if keyboard.is_pressed('e'): dx[2] -= ik_step_size

            if keyboard.is_pressed('space'): current_action[5] += gripper_step 
            if keyboard.is_pressed('c'):     current_action[5] -= gripper_step 

            # Apply Damped Least Squares Inverse Kinematics
            if np.linalg.norm(dx) > 0:
                jacp = np.zeros((3, env.model.nv))
                mujoco.mj_jacSite(env.model, env.data, jacp, None, site_id)
                
                J_arm = jacp[:, :5]
                
                # DLS Calculation: J.T @ inv(J @ J.T + damping^2 * I)
                lambda_sq = damping ** 2
                J_dls = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + lambda_sq * np.eye(3))
                
                delta_q = J_dls @ dx
                
                # Calculate the new target
                new_target = current_action[:5] + delta_q
                
                # ANTI-WINDUP: Prevent target from running away from current physical position
                actual_q = env.data.qpos[:5]
                new_target = np.clip(new_target, actual_q - max_lead, actual_q + max_lead)
                
                current_action[:5] = new_target

            # Enforce hard joint limits
            current_action = np.clip(current_action, lower_limits, upper_limits)

            # Step physics and sync viewer
            env.step(current_action)
            viewer.sync()

    env.close()

if __name__ == "__main__":
    main()