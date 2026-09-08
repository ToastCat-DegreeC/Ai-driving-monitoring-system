import os
import sys
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

def test_single_video(video_path, feature_extractor, enhancement_network, fusion_model):
    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"

    cap = cv2.VideoCapture(video_path)
    rgb_means_buffer = deque(maxlen=config.RPPG_WINDOW_SIZE)
    ear_values_buffer = deque(maxlen=config.EAR_WINDOW_SIZE)
    fusion_history = deque(maxlen=config.FUSION_SEQUENCE_LEN)
    
    frame_count = 0
    max_test_frames = 400 # Increased to allow for 60-frame calibration
    hr_readings = []
    perclos_readings = []
    fatigue_statuses = []
    
    cal_n, ear_s, ear_off = 0, 0.0, 0.3

    while frame_count < max_test_frames:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        landmarks, forehead_roi, cheek_roi = feature_extractor.detect_and_get_landmarks(frame)
        if landmarks:
            rgb_means_buffer.append(feature_extractor.get_roi_means(frame, landmarks, forehead_roi, cheek_roi))
            
            # EAR Calibration & Normalization
            raw_ear = feature_extractor.get_ear_value(frame, landmarks)
            if cal_n < 60:
                ear_s += raw_ear
                cal_n += 1
                if cal_n == 60: ear_off = ear_s / 60.0
                continue # Skip processing until calibrated
            
            norm_ear = (raw_ear / ear_off) * 0.3 if ear_off > 0.01 else raw_ear
            ear_values_buffer.append(np.clip(norm_ear, 0.0, 0.4))

            if frame_count % 30 == 0:
                if len(rgb_means_buffer) == config.RPPG_WINDOW_SIZE:
                    rppg_sig = feature_extractor.calc_rppg_signal(list(rgb_means_buffer))
                    rppg_prep = pad_or_truncate_signal(rppg_sig, config.HR_MODEL_INPUT_LENGTH)
                    ear_prep = pad_or_truncate_signal(list(ear_values_buffer), config.PERCLOS_MODEL_INPUT_LENGTH)
                    
                    raw_hr, perclos = enhancement_network.enhance_features(rppg_prep, ear_prep)
                    fusion_history.append([raw_hr, perclos])
                    
                    # FusionModel.predict_fatigue now handles normalization internally
                    status_text = fusion_model.predict_fatigue(list(fusion_history))
                    
                    hr_readings.append(raw_hr)
                    perclos_readings.append(perclos)
                    fatigue_statuses.append(status_text)

    cap.release()
    if not hr_readings:
        return "No readings captured."
        
    avg_hr = np.mean(hr_readings)
    avg_perclos = np.mean(perclos_readings)
    last_status = fatigue_statuses[-1] if fatigue_statuses else "N/A"
    
    return f"Avg HR: {avg_hr:.1f}, Avg PERCLOS: {avg_perclos:.3f}, Final Status: {last_status}"

def main():
    print("--- Batch Video Testing ---")
    dataset_root = "Data/UBFC-rPPG/DATASET_1"
    subjects = ["5-gt", "6-gt", "7-gt", "8-gt", "10-gt", "11-gt", "12-gt"]
    
    try:
        fe = FeatureExtractor()
        en = EnhancementNetwork(config.MODEL_PATH_HR, config.MODEL_PATH_PERCLOS, config.MODEL_PATH_SPO2)
        fm = FusionModel(config.MODEL_PATH_FUSION)
        print("Models initialized successfully.\n")
    except Exception as e:
        print(f"Init Error: {e}")
        return

    for subject in subjects:
        video_path = os.path.join(dataset_root, subject, "vid.avi")
        print(f"Testing Subject: {subject}...")
        result = test_single_video(video_path, fe, en, fm)
        print(f"Result: {result}\n")

if __name__ == "__main__":
    main()
