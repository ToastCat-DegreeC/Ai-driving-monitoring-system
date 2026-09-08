import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np
from collections import deque
from part_a.feature_extractor import FeatureExtractor
from part_a.cnn_model import build_eye_cnn
from part_b.enhancement_network import EnhancementNetwork
from part_c.fusion_model import FusionModel
from utils.signal_processing import pad_or_truncate_signal
import config

def get_root_path(rel_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', rel_path))

def main():
    print("Verifying Integrated System with Video File...")

    # 1. Initialize all components
    try:
        feature_extractor = FeatureExtractor()
        enhancement_network = EnhancementNetwork(
            weights_path_hr=get_root_path(config.MODEL_PATH_HR), 
            weights_path_perclos=get_root_path(config.MODEL_PATH_PERCLOS),
            weights_path_spo2=get_root_path(config.MODEL_PATH_SPO2)
        )
        fusion_model = FusionModel(get_root_path(config.MODEL_PATH_FUSION))
        eye_cnn = build_eye_cnn()
        eye_weights = get_root_path(config.MODEL_PATH_EYE_CNN)
        if os.path.exists(eye_weights):
            eye_cnn.load_weights(eye_weights)
            print("Loaded Eye CNN weights.")
        
        print("All components initialized successfully.")
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # 2. Load a sample video from the dataset
    video_path = get_root_path("Data/UBFC-rPPG/DATASET_1/5-gt/vid.avi")
    if not os.path.exists(video_path):
        print(f"Test video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    
    # 3. Setup buffers (same as main.py)
    rgb_means_buffer = deque(maxlen=config.RPPG_WINDOW_SIZE)
    ear_values_buffer = deque(maxlen=config.EAR_WINDOW_SIZE)
    fusion_history = deque(maxlen=config.FUSION_SEQUENCE_LEN)
    
    last_heart_rate = config.HR_STABLE_INITIAL
    last_perclos = 0.0
    last_pitch, last_yaw, last_roll = 0.0, 0.0, 0.0
    
    frame_count = 0
    max_test_frames = 300 # Test approx 10 seconds of video
    
    print(f"Starting processing loop for {max_test_frames} frames...")

    while frame_count < max_test_frames:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        
        # Landmarks & ROIs
        landmarks, forehead_roi, cheek_roi = feature_extractor.detect_and_get_landmarks(frame)
        
        if landmarks:
            # Feature Extraction
            current_rgb_means = feature_extractor.get_roi_means(frame, landmarks, forehead_roi, cheek_roi)
            rgb_means_buffer.append(current_rgb_means)
            
            current_ear = feature_extractor.get_ear_value(frame, landmarks)
            ear_values_buffer.append(current_ear)
            
            # Head Pose (Testing Robustness changes)
            head_pose_angles, viz_data = feature_extractor.get_head_pose(frame, landmarks)
            if head_pose_angles:
                raw_pitch, raw_yaw, raw_roll = head_pose_angles
                # EMA Smoothing
                alpha = config.HEAD_POSE_SMOOTHING
                last_pitch = (alpha * raw_pitch) + (1 - alpha) * last_pitch
                last_yaw = (alpha * raw_yaw) + (1 - alpha) * last_yaw
                last_roll = (alpha * raw_roll) + (1 - alpha) * last_roll

            # Signal Processing (Every 30 frames)
            if frame_count % 30 == 0:
                if len(rgb_means_buffer) == config.RPPG_WINDOW_SIZE and len(ear_values_buffer) == config.EAR_WINDOW_SIZE:
                    
                    rppg_sig = feature_extractor.calc_rppg_signal(list(rgb_means_buffer))
                    rppg_prepared = pad_or_truncate_signal(rppg_sig, config.HR_MODEL_INPUT_LENGTH)
                    ear_prepared = pad_or_truncate_signal(list(ear_values_buffer), config.PERCLOS_MODEL_INPUT_LENGTH)
                    
                    raw_hr, perclos = enhancement_network.enhance_features(rppg_prepared, ear_prepared)
                    last_heart_rate = raw_hr
                    last_perclos = perclos
                    
                    # Fusion Logic (Testing Sliding Window)
                    fusion_history.append([last_heart_rate, last_perclos])
                    fatigue_status = fusion_model.predict_fatigue(list(fusion_history))
                    
                    print(f"Frame {frame_count}: HR={last_heart_rate:.1f}, PERCLOS={last_perclos:.3f}, Pose(P/Y)={last_pitch:.1f}/{last_yaw:.1f}, Status={fatigue_status}")

        if frame_count % 100 == 0:
            print(f"Progress: {frame_count} frames processed...")

    cap.release()
    print("\nIntegrated System Verification Complete.")
    print("If you see heart rate values and fatigue status above, the data flow is working correctly!")

if __name__ == "__main__":
    main()
