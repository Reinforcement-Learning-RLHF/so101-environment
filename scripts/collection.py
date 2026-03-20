import datetime
import time
import os
import numpy as np
import mujoco
import mujoco.viewer
import keyboard
import h5py
from envs.arm_env import ArmEnv

def save_episode(dataset_dir, episode_idx, obs_history, action_history):
    """Saves the buffered trajectory into an HDF5 file formatted for ACT."""
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    
    with h5py.File(dataset_path, 'w') as root:
        root.attrs['sim'] = True
        
        obs_group = root.create_group('observations')
        image_group = obs_group.create_group('images')
        
        qpos_data = np.array([obs['qpos'] for obs in obs_history])
        qvel_data = np.array([obs['qvel'] for obs in obs_history])
        action_data = np.array(action_history)
        
        obs_group.create_dataset('qpos', data=qpos_data)
        obs_group.create_dataset('qvel', data=qvel_data)
        
        image_main_data = np.array([obs['image_main'] for obs in obs_history])
        image_wrist_data = np.array([obs['image_wrist'] for obs in obs_history])
        
        image_group.create_dataset('main_observation', data=image_main_data, compression='gzip')
        image_group.create_dataset('wrist_cam', data=image_wrist_data, compression='gzip')
        
        root.create_dataset('action', data=action_data)
        
    print(f"✅ Saved {len(action_history)} steps to {dataset_path}")

def main():
    # CRITICAL: render_images=True so the cameras actually save data!
    env = ArmEnv(render_images=True) 
    obs = env.reset()

    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')

    # Actuator limits from so101.xml
    lower_limits = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
    upper_limits = np.array([ 1.91986,  1.74533,  1.69,  1.65806,  2.84121,  1.74533])

    target_q = np.copy(env.data.qpos[:6])

    # --- TUNING PARAMETERS ---
    control_hz = 60
    base_ik_step = 0.015               
    base_pitch_step = 0.1              
    base_roll_step = 0.1               
    base_gripper = 0.10                 
    damping = 0.05                      
    max_lead = 0.05                     

    # --- DATA COLLECTION SETUP ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = f"./act_dataset_{timestamp}"
    episode_idx = 0
    obs_history = []
    action_history = []

    print("=== SO101 Teleop & Data Collection ===")
    print(" [POSITION] W/S (X), A/D (Y), Q/E (Z)")
    print(" [PITCH]    I/K")
    print(" [ROLL]     U/O")
    print(" [GRIPPER]  SPACE (Close), C (Open)")
    print(" [MODIFIER] Hold SHIFT for ultra-precision mode")
    print(" ------------------------------------")
    print(" [ENTER]    SAVE successful pour to HDF5 and Reset")
    print(" [R]        DISCARD failed pour and Reset")
    print(" ESC to Quit")
    print("======================================")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            if keyboard.is_pressed('esc'):
                break

            # ==========================================
            # 1. HANDLE SAVING AND RESETTING
            # ==========================================
            if keyboard.is_pressed('enter'):
                if len(action_history) > 0:
                    save_episode(dataset_dir, episode_idx, obs_history, action_history)
                    episode_idx += 1
                
                print("Resetting for next episode...")
                obs = env.reset()
                target_q = np.copy(env.data.qpos[:6])
                obs_history.clear()
                action_history.clear()
                viewer.sync()
                time.sleep(0.5) # Debounce
                continue

            if keyboard.is_pressed('r'):
                print("Discarding failed pour... Resetting.")
                obs = env.reset()
                target_q = np.copy(env.data.qpos[:6])
                obs_history.clear()
                action_history.clear()
                viewer.sync()
                time.sleep(0.5)
                continue 

            # ==========================================
            # 2. READ KEYBOARD INPUTS
            # ==========================================
            speed_mult = 0.2 if keyboard.is_pressed('shift') else 1.0

            dx = np.zeros(3) 
            pitch_cmd = 0.0
            roll_cmd = 0.0
            gripper_cmd = 0.0
            
            # Position Inputs (Cartesian)
            if keyboard.is_pressed('w'): dx[0] += base_ik_step * speed_mult
            if keyboard.is_pressed('s'): dx[0] -= base_ik_step * speed_mult
            if keyboard.is_pressed('a'): dx[1] += base_ik_step * speed_mult
            if keyboard.is_pressed('d'): dx[1] -= base_ik_step * speed_mult
            if keyboard.is_pressed('q'): dx[2] += base_ik_step * speed_mult
            if keyboard.is_pressed('e'): dx[2] -= base_ik_step * speed_mult

            # Pitch, Roll, Gripper Inputs
            if keyboard.is_pressed('i'): pitch_cmd -= base_pitch_step * speed_mult
            if keyboard.is_pressed('k'): pitch_cmd += base_pitch_step * speed_mult
            if keyboard.is_pressed('u'): roll_cmd += base_roll_step * speed_mult
            if keyboard.is_pressed('o'): roll_cmd -= base_roll_step * speed_mult
            if keyboard.is_pressed('space'): gripper_cmd += base_gripper * speed_mult
            if keyboard.is_pressed('c'):     gripper_cmd -= base_gripper * speed_mult

            # ==========================================
            # 3. CALCULATE INVERSE KINEMATICS
            # ==========================================
            norm_dx = np.linalg.norm(dx)
            if norm_dx > 0:
                jacp = np.zeros((3, env.model.nv))
                mujoco.mj_jacSite(env.model, env.data, jacp, None, site_id)
                
                J_pos = jacp[:, :3] 
                lambda_sq = damping ** 2
                
                J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda_sq * np.eye(3))
                delta_q = J_dls @ dx
                
                target_q[:3] += delta_q

                # Anti-Windup
                actual_q = env.data.qpos[:3]
                target_q[:3] = np.clip(target_q[:3], actual_q - max_lead, actual_q + max_lead)

            target_q[3] += pitch_cmd
            target_q[4] += roll_cmd
            target_q[5] += gripper_cmd
            target_q = np.clip(target_q, lower_limits, upper_limits)

            # ==========================================
            # 4. RECORD DATA AND STEP PHYSICS
            # ==========================================
            # Map the observation at time T to the action chosen at time T
            obs_history.append(obs)
            action_history.append(np.copy(target_q))

            # Step the environment to get observation for time T+1
            obs, reward, done, info = env.step(target_q)
            viewer.sync()
            
            # Cap framerate to maintain consistent 60Hz dt for the model
            time_until_next_step = (1.0 / control_hz) - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    env.close()

if __name__ == "__main__":
    main()