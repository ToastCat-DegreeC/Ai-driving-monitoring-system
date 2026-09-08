import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np
import time
import tensorflow as tf
from collections import deque
from part_a.feature_extractor import FeatureExtractor
from part_a.cnn_model import build_eye_cnn
from part_b.enhancement_network import EnhancementNetwork
from part_c.fusion_model import FusionModel
from utils.signal_processing import pad_or_truncate_signal
import config

import threading
from queue import Queue

def get_root_path(rel_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', rel_path))

class AIBackgroundWorker(threading.Thread):
    def __init__(self, feature_extractor, enhancement_net, fusion_model):
        super().__init__(daemon=True)
        self.fe = feature_extractor
        self.en = enhancement_net
        self.fm = fusion_model
        self.input_queue = Queue(maxsize=1)
        self.results = {
            'hr': config.HR_STABLE_INITIAL,
            'perclos': 0.0,
            'spo2': 98.0,
            'status': "Calibrating...",
            'new_data': False
        }
        self.running = True
        self.fusion_history = deque(maxlen=config.FUSION_SEQUENCE_LEN)
        self.stable_hr = config.HR_STABLE_INITIAL
        self.hr_history = deque(maxlen=config.HR_HISTORY_LEN)

    def run(self):
        # Force CPU device for the background thread
        with tf.device('/cpu:0'):
            while self.running:
                try:
                    task_data = self.input_queue.get(timeout=0.1)
                    if task_data is None: continue
                    rgb_means = task_data['rgb_means']
                    ear_values = task_data['ear_values']
                    
                    rppg_sig = self.fe.calc_rppg_signal(rgb_means)
                    rppg_prep = pad_or_truncate_signal(rppg_sig, config.HR_MODEL_INPUT_LENGTH)
                    ear_prep = pad_or_truncate_signal(ear_values, config.PERCLOS_MODEL_INPUT_LENGTH)
                    
                    raw_hr, perclos = self.en.enhance_features(rppg_prep, ear_prep)
                    spo2 = self.en.predict_spo2(rgb_means)
                    
                    if config.MIN_HR <= raw_hr <= config.MAX_HR:
                        diff = abs(raw_hr - self.stable_hr)
                        if diff <= config.MAX_HR_JUMP:
                            self.stable_hr = (config.HR_ALPHA * raw_hr) + (1 - config.HR_ALPHA) * self.stable_hr
                        else:
                            self.hr_history.append(raw_hr)
                            if len(self.hr_history) >= 3:
                                avg_h = sum(list(self.hr_history)[-3:]) / 3.0
                                if abs(avg_h - raw_hr) < 5.0:
                                    self.stable_hr = (config.HR_ALPHA * raw_hr) + (1 - config.HR_ALPHA) * self.stable_hr
                    
                    self.fusion_history.append([self.stable_hr, perclos])
                    status = self.fm.predict_fatigue(list(self.fusion_history))
                    
                    self.results['hr'] = self.stable_hr
                    self.results['perclos'] = perclos
                    self.results['spo2'] = spo2
                    self.results['status'] = status
                    self.results['new_data'] = True
                    self.input_queue.task_done()
                except Exception as e:
                    # Ignore timeout-related queue errors
                    if "Empty" not in str(type(e)):
                        print(f"WORKER ERROR: {e}")
                        import traceback
                        traceback.print_exc()

    def stop(self):
        self.running = False

def main():
    print("Verifying Asynchronous System with Video File...")

    feature_extractor = FeatureExtractor()
    enhancement_network = EnhancementNetwork(
        weights_path_hr=get_root_path(config.MODEL_PATH_HR), 
        weights_path_perclos=get_root_path(config.MODEL_PATH_PERCLOS),
        weights_path_spo2=get_root_path(config.MODEL_PATH_SPO2)
    )
    fusion_model = FusionModel(get_root_path(config.MODEL_PATH_FUSION))

    worker = AIBackgroundWorker(feature_extractor, enhancement_network, fusion_model)
    worker.start()
    print("Background Worker started.")

    video_path = get_root_path("Data/UBFC-rPPG/DATASET_1/5-gt/vid.avi")
    cap = cv2.VideoCapture(video_path)
    
    rgb_means_buffer = deque(maxlen=config.RPPG_WINDOW_SIZE)
    ear_values_buffer = deque(maxlen=config.EAR_WINDOW_SIZE)

    frame_count = 0
    max_test_frames = 600
    
    while frame_count < max_test_frames:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        landmarks, forehead_roi, cheek_roi = feature_extractor.detect_and_get_landmarks(frame)
        if landmarks:
            rgb_means_buffer.append(feature_extractor.get_roi_means(frame, landmarks, forehead_roi, cheek_roi))
            ear_values_buffer.append(feature_extractor.get_ear_value(frame, landmarks))

        # Async Trigger (Every 30 frames)
        if frame_count % 30 == 0:
            if len(rgb_means_buffer) == config.RPPG_WINDOW_SIZE:
                if worker.input_queue.empty():
                    print(f"Frame {frame_count}: Offloading batch to background thread...")
                    worker.input_queue.put({
                        'rgb_means': list(rgb_means_buffer),
                        'ear_values': list(ear_values_buffer)
                    })

        # Check for results
        if worker.results['new_data']:
            print(f"--- Worker Update at frame {frame_count} ---")
            print(f"    HR: {worker.results['hr']:.1f}")
            print(f"    SpO2: {worker.results['spo2']:.1f}%")
            print(f"    Status: {worker.results['status']}")
            worker.results['new_data'] = False

        if frame_count % 100 == 0:
            print(f"Main Loop Progress: {frame_count} frames...")

    worker.stop()
    cap.release()
    print("\nAsync System Verification Complete.")

if __name__ == "__main__":
    main()
