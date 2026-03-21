import h5py
import cv2
import numpy as np
import os

def replay_episode(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    print(f"Loading {file_path}...")
    
    with h5py.File(file_path, 'r') as f:
        # Load the image arrays into memory
        main_imgs = f['observations/images/main_observation'][:]
        wrist_imgs = f['observations/images/wrist_cam'][:]
        actions = f['action'][:]
        
    num_frames = len(main_imgs)
    print(f"Loaded {num_frames} frames. Press 'Q' to exit.")

    for i in range(num_frames):
        # 1. MuJoCo saved them as (Channels, Height, Width) normalized to [0, 1].
        # OpenCV needs them as (Height, Width, Channels) scaled to [0, 255].
        main_frame = (np.transpose(main_imgs[i], (1, 2, 0)) * 255).astype(np.uint8)
        wrist_frame = (np.transpose(wrist_imgs[i], (1, 2, 0)) * 255).astype(np.uint8)
        
        # 2. MuJoCo renders in RGB, but OpenCV displays in BGR.
        main_frame = cv2.cvtColor(main_frame, cv2.COLOR_RGB2BGR)
        wrist_frame = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)
        
        # 3. Add text to see the frame number
        cv2.putText(main_frame, f"Frame: {i}/{num_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 4. Show the images side-by-side
        cv2.imshow('Main Camera - Replay', main_frame)
        cv2.imshow('Wrist Camera - Replay', wrist_frame)
        
        # Wait 16ms to approximate 60 FPS playback. Press 'q' to quit early.
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change this to whatever episode number you want to check
    target_file = "act_dataset_20260321_161742/episode_12.hdf5"
    replay_episode(target_file)