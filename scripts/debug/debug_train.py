import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import csv
import numpy as np
import tensorflow as tf
from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
from utils.video_processing import read_frames_from_video 
from utils.signal_processing import pad_or_truncate_signal

# --- 1. CONFIGURATION ---
DMD_DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Data"))
UBFC_DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Data/UBFC-rPPG/DATASET_1"))
HR_MODEL_WEIGHTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/hr_model_debug.h5"))
PERCLOS_MODEL_WEIGHTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/perclos_model_debug.h5"))

# Training Hyperparameters
EPOCHS = 1
BATCH_SIZE = 2
WINDOW_SIZE_SECS = 5 # Reduced for speed
FPS = 30 

def parse_ubfc_dataset(dataset_root):
    """
    Parses the UBFC-rPPG dataset (DATASET_1) to get video paths and ground truth PPG.
    """
    print(f"Parsing UBFC dataset from: {dataset_root}")
    parsed_data = []

    if not os.path.exists(dataset_root):
        print(f"Error: UBFC dataset root '{dataset_root}' does not exist.")
        return parsed_data

    # Iterate through subject folders (e.g., '1-gt', '2-gt')
    count = 0
    for subject_folder in os.listdir(dataset_root):
        # if count >= 1: break # LIMIT TO 1 SUBJECT FOR DEBUGGING
        
        subject_path = os.path.join(dataset_root, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        video_path = os.path.join(subject_path, "vid.avi")
        gt_path = os.path.join(subject_path, "gtdump.xmp")

        if os.path.exists(video_path) and os.path.exists(gt_path):
            try:
                # Parse gtdump.xmp (CSV format: Time, HR, SpO2, PPG)
                ppg_signal = []
                with open(gt_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 4:
                            ppg_signal.append(float(row[3])) # 4th column is PPG
                
                ppg_signal = np.array(ppg_signal)
                parsed_data.append((video_path, ppg_signal))
                print(f"Parsed {video_path}, Signal length: {len(ppg_signal)}")
                count += 1
            except Exception as e:
                print(f"Error parsing {gt_path}: {e}")

    return parsed_data

def preprocess_data(parsed_data, feature_extractor, model_type):
    """
    Pre-processes all data into memory to avoid generator issues.
    """
    X_all = []
    Y_all = []
    
    RPPG_INPUT_LENGTH = 256
    
    print(f"Pre-processing {len(parsed_data)} samples for {model_type}...")
    
    for i, data_item in enumerate(parsed_data):
        if model_type == 'HR':
            video_path, gt_ppg = data_item
            print(f"  Processing video {i+1}/{len(parsed_data)}: {video_path}")
            
            try:
                frames = read_frames_from_video(video_path, max_frames=WINDOW_SIZE_SECS * FPS)
                if not frames: 
                    print(f"  Warning: No frames read from {video_path}")
                    continue
                
                print(f"    Read {len(frames)} frames. Extracting rPPG...")
                rppg_signal = feature_extractor.extract_rppg(frames)
                print(f"    Extracted rPPG signal length: {len(rppg_signal)}")
                
                if len(rppg_signal) == 0:
                     print("    ERROR: rPPG signal is empty.")
                     continue

                rppg_processed = pad_or_truncate_signal(rppg_signal, RPPG_INPUT_LENGTH)
                
                # Check for NaNs
                if np.isnan(rppg_processed).any():
                    print("    WARNING: NaNs found in processed signal. Replacing with zeros.")
                    rppg_processed = np.nan_to_num(rppg_processed)

                X_all.append(rppg_processed.reshape(-1, 1))
                
                # Calculate a simple scalar HR from the GT signal for the label
                peaks = 0 
                # Simple peak detection for debugging
                gt_ppg_segment = gt_ppg[:len(frames)] # Match duration
                for p in range(1, len(gt_ppg_segment)-1):
                    if gt_ppg_segment[p] > gt_ppg_segment[p-1] and gt_ppg_segment[p] > gt_ppg_segment[p+1]:
                        peaks +=1
                duration_sec = len(gt_ppg_segment) / 30.0 
                hr_bpm = (peaks / duration_sec) * 60 if duration_sec > 0 else 70
                print(f"    Calculated GT HR: {hr_bpm} bpm")
                Y_all.append(hr_bpm)
            
            except Exception as e:
                print(f"    EXCEPTION during processing: {e}")
                import traceback
                traceback.print_exc()

    return np.array(X_all), np.array(Y_all)

def main():
    print("--- Loading Datasets ---")
    ubfc_data = parse_ubfc_dataset(UBFC_DATASET_ROOT)
    
    if not ubfc_data:
        print("No UBFC data found to debug.")
        return

    # No split, just use same data
    train_hr_raw = ubfc_data
    
    feature_extractor = FeatureExtractor(rppg_method='POS') 
    enhancement_net = EnhancementNetwork()
    
    print("\n--- Pre-processing HR Training Data ---")
    X_train_hr, Y_train_hr = preprocess_data(train_hr_raw, feature_extractor, 'HR')

    # --- 4. Train the Heart Rate Model ---
    if len(X_train_hr) > 0:
        print("\n--- Training Heart Rate Model (on UBFC-rPPG) ---")
        try:
            enhancement_net.hr_model.fit(
                x=X_train_hr, y=Y_train_hr,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS
            )
        except Exception as e:
            print(f"Training failed: {e}")
    else:
        print("Skipping HR training (No valid data processed).")

if __name__ == '__main__':
    main()
