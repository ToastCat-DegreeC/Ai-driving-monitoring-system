import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np
from part_a.feature_extractor import FeatureExtractor
import config

def main():
    print("Debugging EAR values for subject 5-gt...")
    dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Data/UBFC-rPPG/DATASET_1/5-gt"))
    video_path = os.path.join(dataset_root, "vid.avi")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    fe = FeatureExtractor()
    cap = cv2.VideoCapture(video_path)
    
    ear_values = []
    frame_count = 0
    max_frames = 100
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        landmarks, _, _ = fe.detect_and_get_landmarks(frame)
        if landmarks:
            ear = fe.get_ear_value(frame, landmarks)
            ear_values.append(ear)
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}: EAR = {ear:.4f}")

    cap.release()
    
    if ear_values:
        print(f"\nEAR Summary for {frame_count} frames:")
        print(f"  Min: {np.min(ear_values):.4f}")
        print(f"  Max: {np.max(ear_values):.4f}")
        print(f"  Mean: {np.mean(ear_values):.4f}")
        print(f"  Threshold for 'Closed' in synthetic training: 0.2")
    else:
        print("No landmarks detected.")

if __name__ == "__main__":
    main()
