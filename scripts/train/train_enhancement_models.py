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
DMD_DATASET_ROOT = os.path.join(os.path.dirname(__file__), "../../Data") 
UBFC_DATASET_ROOT = os.path.join(os.path.dirname(__file__), "../../Data/UBFC-rPPG/DATASET_1")
HR_MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "../../models/hr_model.h5")
PERCLOS_MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "../../models/perclos_model.h5")

# Training Hyperparameters
EPOCHS = 20
BATCH_SIZE = 4 
LEARNING_RATE = 1e-3
WINDOW_SIZE_SECS = 10 
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
    for subject_folder in os.listdir(dataset_root):
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
            except Exception as e:
                print(f"Error parsing {gt_path}: {e}")

    return parsed_data

def parse_dmd_dataset(dataset_root):
    """
    Parses the Vicomtech DMD dataset to get video paths and ground truth.
    """
    print(f"Parsing DMD dataset from: {dataset_root}")
    
    parsed_data = []
    
    # Define mapping for eye states for PERCLOS
    eye_state_mapping = {
        "Closed": 0, "Closing": 0, "Openend": 1, "Opening": 1
    }

    if not os.path.exists(dataset_root):
        print(f"Warning: Dataset root {dataset_root} not found.")
        return []

    # Iterate through groups (gA, gB, etc.)
    for group_name in sorted(os.listdir(dataset_root)): 
        group_path = os.path.join(dataset_root, group_name)
        if not os.path.isdir(group_path) or not group_name.startswith('g'):
            continue

        # Iterate through subjects (sub1, sub2, etc.)
        for subject_name in sorted(os.listdir(group_path)): 
            subject_path = os.path.join(group_path, subject_name)
            if not os.path.isdir(subject_path) or not subject_name.startswith('sub'):
                continue
            
            # Focus on 's5' for drowsiness data
            session_name = 's5'
            session_path = os.path.join(subject_path, session_name)
            if not os.path.isdir(session_path):
                continue 

            # Find mosaic video and its corresponding JSON
            for entry in sorted(os.listdir(session_path)): 
                if entry.endswith('_mosaic.mp4'):
                    video_file = entry
                    json_file = video_file.replace('.mp4', '.json')
                    
                    video_path = os.path.join(session_path, video_file)
                    json_path = os.path.join(session_path, json_file)

                    if not os.path.exists(json_path):
                        continue
                    
                    try:
                        with open(json_path, 'r') as f:
                            annotation_data = json.load(f)
                    except Exception as e:
                        continue

                    num_frames_annotated = len(annotation_data.get('frames', []))
                    if num_frames_annotated == 0:
                        continue
                    
                    ground_truth_eye_states = np.full(num_frames_annotated, 1, dtype=int) 
                    
                    action_definitions = {action['id']: action['type'] for action in annotation_data.get('actions', [])}
                    
                    for action_id, action_type in action_definitions.items():
                        if action_type in eye_state_mapping:
                            mapped_value = eye_state_mapping[action_type]
                            for action_obj in annotation_data.get('actions', []):
                                if action_obj['id'] == action_id and 'frame_intervals' in action_obj:
                                    for interval in action_obj['frame_intervals']:
                                        start_frame = interval[0]
                                        end_frame = interval[1] 
                                        for frame_idx in range(start_frame, min(end_frame + 1, num_frames_annotated)):
                                            ground_truth_eye_states[frame_idx] = mapped_value
                                    break 

                    ground_truth_hr_signal = np.zeros(num_frames_annotated) 

                    parsed_data.append((video_path, ground_truth_hr_signal, ground_truth_eye_states))
                    print(f"Parsed {video_path} with {num_frames_annotated} frames.")

    if not parsed_data:
        print(f"WARNING: No s5 session data or valid annotation files found in '{dataset_root}'.")

    return parsed_data

def preprocess_data(parsed_data, feature_extractor, model_type):
    """
    Pre-processes all data into memory to avoid generator issues.
    """
    X_all = []
    Y_all = []
    
    RPPG_INPUT_LENGTH = 256
    EAR_INPUT_LENGTH = 128
    
    print(f"Pre-processing {len(parsed_data)} samples for {model_type}...")
    
    for i, data_item in enumerate(parsed_data):
        if model_type == 'HR':
            video_path, gt_ppg = data_item
            print(f"  Processing video {i+1}/{len(parsed_data)}: {video_path}")
            
            frames = read_frames_from_video(video_path, max_frames=WINDOW_SIZE_SECS * FPS)
            if not frames: 
                print(f"  Warning: No frames read from {video_path}")
                continue

            rppg_signal = feature_extractor.extract_rppg(frames)
            
            rppg_processed = pad_or_truncate_signal(rppg_signal, RPPG_INPUT_LENGTH)
            
            X_all.append(rppg_processed.reshape(-1, 1))
            
            # Calculate a simple scalar HR from the GT signal for the label
            peaks = 0 
            for p in range(1, len(gt_ppg)-1):
                if gt_ppg[p] > gt_ppg[p-1] and gt_ppg[p] > gt_ppg[p+1]:
                    peaks +=1
            duration_sec = len(gt_ppg) / 30.0 
            hr_bpm = (peaks / duration_sec) * 60 if duration_sec > 0 else 70
            Y_all.append(hr_bpm)

        elif model_type == 'PERCLOS':
            video_path, _, gt_eye_states = data_item
            print(f"  Processing video {i+1}/{len(parsed_data)}: {video_path}")

            frames = read_frames_from_video(video_path, max_frames=WINDOW_SIZE_SECS * FPS)
            if not frames: continue

            ear_series = feature_extractor.extract_ear_time_series(frames)
            
            ear_processed = pad_or_truncate_signal(ear_series, EAR_INPUT_LENGTH)
            X_all.append(ear_processed.reshape(-1, 1))
            
            if len(gt_eye_states) > 0:
                perclos = np.sum(gt_eye_states == 0) / len(gt_eye_states) # 0 is closed
            else:
                perclos = 0.0
            Y_all.append(perclos)
            
    return np.array(X_all), np.array(Y_all)

def main():
    """Main training function."""
    
    # --- 1. Load and Parse Data ---
    print("--- Loading Datasets ---")
    ubfc_data = parse_ubfc_dataset(UBFC_DATASET_ROOT)
    dmd_data = parse_dmd_dataset(DMD_DATASET_ROOT)
    
    # Split UBFC data (HR)
    split_hr = int(len(ubfc_data) * 0.8)
    train_hr_raw = ubfc_data[:split_hr]
    val_hr_raw = ubfc_data[split_hr:]

    # Split DMD data (PERCLOS)
    split_perclos = int(len(dmd_data) * 0.8)
    train_perclos_raw = dmd_data[:split_perclos]
    val_perclos_raw = dmd_data[split_perclos:]
    
    # --- 2. Instantiate Feature Extractor and Enhancement Network ---
    feature_extractor = FeatureExtractor(rppg_method='POS') 
    enhancement_net = EnhancementNetwork()
    
    # --- 3. Pre-process Data into Memory ---
    # HR Data
    if train_hr_raw:
        print("\n--- Pre-processing HR Training Data ---")
        X_train_hr, Y_train_hr = preprocess_data(train_hr_raw, feature_extractor, 'HR')
        print("\n--- Pre-processing HR Validation Data ---")
        X_val_hr, Y_val_hr = preprocess_data(val_hr_raw, feature_extractor, 'HR')
    else:
        X_train_hr, Y_train_hr, X_val_hr, Y_val_hr = [], [], [], []

    # PERCLOS Data
    if train_perclos_raw:
        print("\n--- Pre-processing PERCLOS Training Data ---")
        X_train_perclos, Y_train_perclos = preprocess_data(train_perclos_raw, feature_extractor, 'PERCLOS')
        print("\n--- Pre-processing PERCLOS Validation Data ---")
        X_val_perclos, Y_val_perclos = preprocess_data(val_perclos_raw, feature_extractor, 'PERCLOS')
    else:
        X_train_perclos, Y_train_perclos, X_val_perclos, Y_val_perclos = [], [], [], []

    # --- 4. Train the Heart Rate Model ---
    if len(X_train_hr) > 0:
        print("\n--- Training Heart Rate Model (on UBFC-rPPG) ---")
        enhancement_net.hr_model.fit(
            x=X_train_hr, y=Y_train_hr,
            validation_data=(X_val_hr, Y_val_hr) if len(X_val_hr) > 0 else None,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS
        )
    else:
        print("Skipping HR training (No UBFC data found).")
    
    # --- 5. Train the PERCLOS Model ---
    if len(X_train_perclos) > 0:
        print("\n--- Training PERCLOS Model (on DMD) ---")
        enhancement_net.perclos_model.fit(
            x=X_train_perclos, y=Y_train_perclos,
            validation_data=(X_val_perclos, Y_val_perclos) if len(X_val_perclos) > 0 else None,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS
        )
    else:
        print("Skipping PERCLOS training (No DMD data found).")
    
    # --- 6. Save the Trained Weights ---
    print("\n--- Saving Model Weights ---")
    enhancement_net.save_weights(HR_MODEL_WEIGHTS_PATH, PERCLOS_MODEL_WEIGHTS_PATH)
    
    print("Training complete.")

if __name__ == '__main__':
    main()