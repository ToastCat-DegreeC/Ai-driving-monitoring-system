import os
import cv2
import numpy as np
import tqdm
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
import config

class NTHUDataLoader:
    def __init__(self, nthu_root_dir):
        self.root_dir = nthu_root_dir
        self.extractor = FeatureExtractor()
        # We use Part B to estimate physiological features for the behavioral sequences
        self.enhancement = EnhancementNetwork()
        self.seq_len = config.FUSION_SEQUENCE_LEN

    def process_subject_scenario(self, subject_id, scenario):
        """
        Processes a single scenario for a subject.
        Returns: X (sequences), Y (labels)
        """
        video_path = os.path.join(self.root_dir, subject_id, scenario, f"{subject_id}_{scenario}.avi")
        label_path = os.path.join(self.root_dir, subject_id, scenario, f"{subject_id}_{scenario}_drowsiness.txt")

        if not os.path.exists(video_path) or not os.path.exists(label_path):
            print(f"Warning: Missing data for {subject_id} {scenario}")
            return None, None

        # Load labels
        with open(label_path, 'r') as f:
            labels = [int(line.strip()) for line in f.readlines()]

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sync labels and frames (NTHU labels are per frame)
        labels = labels[:frame_count]

        X_data = []
        Y_data = []
        
        # Buffer to hold features for sequence generation
        feature_history = []
        ear_history = []

        print(f"Processing {subject_id} - {scenario} ({frame_count} frames)...")
        
        pbar = tqdm.tqdm(total=frame_count)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Extract Raw Features (Part A)
            landmarks = self.extractor.extract_landmarks(frame)
            if landmarks:
                ear = self.extractor.calculate_ear(landmarks)
                mar = self.extractor.calculate_mar(landmarks)
                pose_var = self.extractor.estimate_head_pose_variance(landmarks)
                
                # Extract rPPG means for Physiological estimation
                roi_means = self.extractor.extract_roi_means(frame, landmarks)
                self.extractor.update_buffers(roi_means)
                ear_history.append(ear)
                if len(ear_history) > config.EAR_WINDOW_SIZE:
                    ear_history.pop(0)
                
                # 2. Get Physiological Estimates (Part B)
                # We wait until we have enough buffer for a stable HR/SpO2 estimate
                if len(self.extractor.rgb_buffers['Forehead']) >= config.RPPG_BUFFER_SIZE:
                    rppg_signal = self.extractor.calc_rppg_signal()
                    hr, perclos, hrv = self.enhancement.enhance_features(rppg_signal, ear_history)
                    spo2 = self.enhancement.predict_spo2(rppg_signal)
                else:
                    hr, perclos, hrv, spo2 = 75.0, 0.0, 50.0, 98.0 # Defaults

                # Calculate HRV Delta (Feature 7)
                hrv_delta = 0.0
                if len(feature_history) > 0:
                    hrv_delta = hrv - feature_history[-1][5]

                # Current feature vector: [HR, PERCLOS, MAR, Pose_Var, SpO2, HRV, HRV_Delta]
                features = [hr, perclos, mar, pose_var, spo2, hrv, hrv_delta]
                feature_history.append(features)
                
                if len(feature_history) >= self.seq_len:
                    # Current label is the label of the last frame in the sequence
                    current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    if current_idx < len(labels):
                        X_data.append(np.array(feature_history[-self.seq_len:]))
                        Y_data.append(labels[current_idx])
            
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        return np.array(X_data), np.array(Y_data)

    def build_dataset(self, subjects, scenarios, output_path):
        """
        Builds a full .npy dataset from multiple subjects and scenarios.
        """
        all_X = []
        all_Y = []
        
        for sub in subjects:
            for sce in scenarios:
                X, Y = self.process_subject_scenario(sub, sce)
                if X is not None:
                    all_X.append(X)
                    all_Y.append(Y)
        
        if all_X:
            final_X = np.concatenate(all_X, axis=0)
            final_Y = np.concatenate(all_Y, axis=0)
            
            np.save(f"{output_path}_X.npy", final_X)
            np.save(f"{output_path}_Y.npy", final_Y)
            print(f"Dataset saved to {output_path}_X.npy and {output_path}_Y.npy")
            print(f"Total samples: {len(final_X)}")
        else:
            print("No data processed.")

if __name__ == "__main__":
    # Example usage (assuming NTHU is at D:/NTHU-DDD/Training)
    # nthu_path = "D:/NTHU-DDD/Training"
    # loader = NTHUDataLoader(nthu_path)
    # loader.build_dataset(["Subject01"], ["Day_BareFace", "Night_BareFace"], "data/nthu_train_subset")
    print("NTHU Data Loader Script Ready. Please update 'nthu_path' and run when dataset is available.")
