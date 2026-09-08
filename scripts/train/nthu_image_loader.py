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

class NTHUImageLoader:
    def __init__(self, nthu_base_dir):
        self.base_dir = nthu_base_dir
        self.extractor = FeatureExtractor()
        self.enhancement = EnhancementNetwork()
        self.seq_len = config.FUSION_SEQUENCE_LEN

    def get_image_groups(self, category_dir):
        """
        Groups images by Subject_Scenario_Behavior.
        Pattern: SubjectID_Scenario_Behavior_FrameNumber_Label.jpg
        """
        groups = {}
        files = [f for f in os.listdir(category_dir) if f.endswith('.jpg')]
        
        for f in files:
            # Example: 001_glasses_yawning_1000_drowsy.jpg
            # Parts: [0]Sub, [1]Sce, [2]Beh, [3]Frame, [4]Label
            parts = f.split('_')
            if len(parts) < 5: continue
            
            group_key = f"{parts[0]}_{parts[1]}_{parts[2]}"
            if group_key not in groups:
                groups[group_key] = []
            
            # Store tuple of (frame_num, full_path)
            try:
                frame_num = int(parts[3])
                groups[group_key].append((frame_num, os.path.join(category_dir, f)))
            except:
                continue
                
        # Sort each group by frame number
        for key in groups:
            groups[key].sort(key=lambda x: x[0])
            
        return groups

    def process_group(self, group_images, label_val):
        """
        Processes a sequence of images.
        label_val: 1.0 for Drowsy, 0.0 for Not Drowsy
        """
        X_data = []
        Y_data = []
        feature_history = []
        
        # Reset extractor buffers for each new sequence to avoid cross-subject signal bleeding
        self.extractor.rgb_buffers = {roi: [] for roi in ['Forehead', 'L-Cheek', 'R-Cheek']}

        for _, img_path in group_images:
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            landmarks = self.extractor.extract_landmarks(frame)
            if landmarks:
                ear = self.extractor.calculate_ear(landmarks)
                mar = self.extractor.calculate_mar(landmarks)
                pose_var = self.extractor.estimate_head_pose_variance(landmarks)
                
                # Physiological estimation
                roi_means = self.extractor.extract_roi_means(frame, landmarks)
                self.extractor.update_buffers(roi_means)
                
                if len(self.extractor.rgb_buffers['Forehead']) >= config.RPPG_BUFFER_SIZE:
                    rppg_signal = self.extractor.calc_rppg_signal()
                    hr, perclos, hrv = self.enhancement.enhance_features(rppg_signal, ear)
                    spo2 = self.enhancement.predict_spo2(rppg_signal)
                else:
                    hr, perclos, hrv, spo2 = 75.0, 0.0, 50.0, 98.0
                
                features = [hr, perclos, mar, pose_var, spo2, hrv]
                feature_history.append(features)
                
                if len(feature_history) >= self.seq_len:
                    X_data.append(np.array(feature_history[-self.seq_len:]))
                    Y_data.append(label_val)
                    
        return X_data, Y_data

    def run(self, output_path):
        all_X = []
        all_Y = []
        
        # Define categories to process
        categories = [
            (os.path.join(self.base_dir, "Multi class/train/drowsy/sleepyCombination"), 1.0),
            (os.path.join(self.base_dir, "Multi class/train/drowsy/slowBlinkWithNodding"), 1.0),
            (os.path.join(self.base_dir, "Multi class/train/drowsy/yawning"), 1.0),
            (os.path.join(self.base_dir, "Multi class/train/notdrowsy"), 0.0)
        ]
        
        for cat_path, label in categories:
            if not os.path.exists(cat_path):
                print(f"Skipping missing category: {cat_path}")
                continue
                
            print(f"\nScanning category: {os.path.basename(cat_path)}")
            groups = self.get_image_groups(cat_path)
            
            for group_key in tqdm.tqdm(groups, desc=f"Processing {os.path.basename(cat_path)}"):
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
            print(f"\nSuccess! Saved {len(X_final)} sequences to {output_path}")
        else:
            print("No data processed.")

if __name__ == "__main__":
    # Path found on user machine
    nthu_dir = "D:/project/datasets/NTHU-Data"
    loader = NTHUImageLoader(nthu_dir)
    # Start with a subset (you can modify this to process everything)
    loader.run("data/nthu_empirical_fusion")
