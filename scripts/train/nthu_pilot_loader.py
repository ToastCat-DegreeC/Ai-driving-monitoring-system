import os
import cv2
import numpy as np
import tqdm
import sys
import re

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
import config

class NTHUPilotLoader:
    def __init__(self, nthu_base_dir):
        self.base_dir = nthu_base_dir
        self.extractor = FeatureExtractor()
        self.enhancement = EnhancementNetwork()
        self.seq_len = config.FUSION_SEQUENCE_LEN

    def get_image_groups(self, category_dir, limit_groups=None):
        """
        Groups images by Subject_Scenario_Behavior.
        Pattern: SubjectID_Scenario_Behavior_FrameNumber_Label.jpg
        """
        groups = {}
        if not os.path.exists(category_dir):
            return groups
            
        files = [f for f in os.listdir(category_dir) if f.endswith('.jpg')]
        
        for f in files:
            parts = f.split('_')
            if len(parts) < 5: continue
            
            group_key = f"{parts[0]}_{parts[1]}_{parts[2]}"
            if group_key not in groups:
                groups[group_key] = []
            
            try:
                # Frame number is part 3
                frame_num = int(parts[3])
                groups[group_key].append((frame_num, os.path.join(category_dir, f)))
            except:
                continue
                
        for key in groups:
            groups[key].sort(key=lambda x: x[0])
            
        if limit_groups:
            keys = list(groups.keys())[:limit_groups]
            return {k: groups[k] for k in keys}
            
        return groups

    def process_group(self, group_images, label_val):
        X_data = []
        Y_data = []
        feature_history = []
        rgb_means_buffer = [] # Local buffer for rPPG
        ear_history = [] # Local buffer for PERCLOS model

        print(f"  Starting group with {len(group_images)} images...")
        for i, (_, img_path) in enumerate(group_images):
            try:
                frame = cv2.imread(img_path)
                if frame is None: 
                    print(f"    Warning: Could not read image {img_path}")
                    continue
                
                landmarks, rois = self.extractor.detect_and_get_landmarks(frame)
                if landmarks:
                    ear = self.extractor.get_ear_value(frame, landmarks)
                    mar = self.extractor.get_mar_value(frame, landmarks)
                    pose, _ = self.extractor.get_head_pose(frame, landmarks)
                    
                    # Estimate variance if we have pose history, else 0
                    pose_var = 0.1 
                    
                    roi_means = self.extractor.get_roi_means(frame, landmarks, rois)
                    rgb_means_buffer.append(roi_means)
                    ear_history.append(ear)
                    
                    if len(rgb_means_buffer) > config.RPPG_BUFFER_SIZE:
                        rgb_means_buffer.pop(0)
                    if len(ear_history) > config.EAR_WINDOW_SIZE:
                        ear_history.pop(0)
                    
                    if len(rgb_means_buffer) >= 64: # Minimum for SQI/rPPG
                        rppg_signal = self.extractor.calc_rppg_signal(rgb_means_buffer)
                        hr, perclos, hrv = self.enhancement.enhance_features(rppg_signal, ear_history)
                        spo2 = self.extractor.calc_spo2(rgb_means_buffer)
                    else:
                        hr, perclos, hrv, spo2 = 75.0, 0.0, 50.0, 98.0

                    # Calculate HRV Delta (Feature 7)
                    hrv_delta = 0.0
                    if len(feature_history) > 0:
                        hrv_delta = hrv - feature_history[-1][5]

                    features = [hr, perclos, mar, pose_var, spo2, hrv, hrv_delta]
                    feature_history.append(features)

                    if len(feature_history) >= self.seq_len:
                        X_data.append(np.array(feature_history[-self.seq_len:]))
                        Y_data.append(label_val)                
                if (i+1) % 50 == 0:
                    print(f"    Processed {i+1}/{len(group_images)} frames...")
            except Exception as e:
                print(f"    Error processing frame {i}: {e}")
                import traceback
                traceback.print_exc()
                raise e # Re-raise to see the full crash
                    
        return X_data, Y_data

    def run(self, output_path):
        all_X = []
        all_Y = []
        
        categories = [
            (os.path.join(self.base_dir, "Multi class/train/drowsy/sleepyCombination"), 1.0),
            (os.path.join(self.base_dir, "Multi class/train/drowsy/slowBlinkWithNodding"), 1.0),
            (os.path.join(self.base_dir, "Multi class/train/drowsy/yawning"), 1.0),
            (os.path.join(self.base_dir, "Multi class/train/notdrowsy"), 0.0)
        ]
        
        for cat_path, label in categories:
            if not os.path.exists(cat_path):
                print(f"Path not found: {cat_path}")
                continue
            
            groups = self.get_image_groups(cat_path, limit_groups=2)
            print(f"\nProcessing {len(groups)} groups in {os.path.basename(cat_path)}: {list(groups.keys())}")
            
            for group_key in groups:
                print(f"  -> Processing group_key: {group_key}")
                X, Y = self.process_group(groups[group_key], label)
                if X:
                    all_X.extend(X)
                    all_Y.extend(Y)
                    
        if all_X:
            X_final = np.array(all_X, dtype=np.float32)
            Y_final = np.array(all_Y, dtype=np.float32)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(f"{output_path}_X.npy", X_final)
            np.save(f"{output_path}_Y.npy", Y_final)
            print(f"\nPilot Complete! Saved {len(X_final)} sequences to {output_path}")
        else:
            print("No data processed. Check dataset paths and file names.")

if __name__ == "__main__":
    nthu_dir = "D:/project/datasets/NTHU-Data"
    loader = NTHUPilotLoader(nthu_dir)
    loader.run("data/nthu_pilot")
