import cv2
import numpy as np
import os
import pandas as pd
import time
from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
import config
from utils.signal_processing import pad_or_truncate_signal, calculate_sdnn

def load_gt_v1(gt_path):
    """
    Loads ground truth from gtdump.xmp (Dataset 1).
    Format: time_ms, hr, spo2, ppg
    """
    try:
        data = pd.read_csv(gt_path, header=None)
        data.columns = ['time_ms', 'hr', 'spo2', 'ppg']
        return data
    except Exception as e:
        print(f"Error loading GT V1 {gt_path}: {e}")
        return None

def load_gt_v2(gt_path):
    """
    Loads ground truth from ground_truth.txt (Dataset 2).
    Line 1: PPG, Line 2: HR, Line 3: Time(s) in scientific notation
    """
    try:
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        ppg = np.fromstring(lines[0], sep=' ')
        hr = np.fromstring(lines[1], sep=' ')
        time_s = np.fromstring(lines[2], sep=' ')
        # SpO2 is not available in Dataset 2, using 98.0 as placeholder
        return pd.DataFrame({'time_ms': time_s * 1000, 'hr': hr, 'spo2': 98.0, 'ppg': ppg})
    except Exception as e:
        print(f"Error loading GT V2 {gt_path}: {e}")
        return None

def benchmark(dataset_paths, use_skin_mask=True):
    fe = FeatureExtractor()
    en = EnhancementNetwork(config.MODEL_PATH_HR, config.MODEL_PATH_PERCLOS, config.MODEL_PATH_SPO2)
    results = []
    
    # 1. Discover all subjects across provided paths
    all_subjects = []
    for path in dataset_paths:
        if not os.path.exists(path): continue
        subs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        all_subjects.extend(subs)
    
    if not all_subjects:
        print("No subject directories found!")
        return None

    print(f"\n>>> Starting Universal Benchmark (Skin Masking: {use_skin_mask}) on {len(all_subjects)} subjects...")
    
    for subject_dir in all_subjects:
        subject_name = os.path.basename(subject_dir)
        video_path = os.path.join(subject_dir, 'vid.avi')
        gt_v1 = os.path.join(subject_dir, 'gtdump.xmp')
        gt_v2 = os.path.join(subject_dir, 'ground_truth.txt')
        
        if not os.path.exists(video_path):
            continue
            
        if os.path.exists(gt_v2):
            gt_data = load_gt_v2(gt_v2)
            version = "V2"
        elif os.path.exists(gt_v1):
            gt_data = load_gt_v1(gt_v1)
            version = "V1"
        else:
            continue
        
        if gt_data is None: continue
        
        print(f"Processing {subject_name} ({version})...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        rgb_means_buffer = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            l, rois = fe.detect_and_get_landmarks(frame)
            if use_skin_mask:
                rgb_means_buffer.append(fe.get_roi_means(frame, l, rois))
            else:
                # Fallback for unmasked baseline
                roi = rois['forehead'] if rois['forehead'] else rois['left_cheek']
                if roi:
                    roi = fe._clip_roi(roi, frame.shape)
                    if roi:
                        x, y, w, h = roi
                        roi_frame = frame[y:y+h, x:x+w]
                        rgb_means_buffer.append({'forehead': np.mean(roi_frame, axis=(0, 1)) if roi_frame.size > 0 else np.zeros(3), 'left_cheek': np.zeros(3), 'right_cheek': np.zeros(3)})
                    else: rgb_means_buffer.append({'forehead': np.zeros(3), 'left_cheek': np.zeros(3), 'right_cheek': np.zeros(3)})
                else: rgb_means_buffer.append({'forehead': np.zeros(3), 'left_cheek': np.zeros(3), 'right_cheek': np.zeros(3)})
                
            if frame_count % 500 == 0:
                print(f"  Frame {frame_count} processed...")
                
        cap.release()
        
        ws, step = config.RPPG_WINDOW_SIZE, 30
        p_hr_list, p_spo2_list, gt_hr_list, gt_spo2_list = [], [], [], []
        
        for i in range(0, len(rgb_means_buffer) - ws, step):
            win = rgb_means_buffer[i : i + ws]
            sig = fe.calc_rppg_signal(win)
            prep = pad_or_truncate_signal(sig, config.HR_MODEL_INPUT_LENGTH)
            
            try:
                prep = en._butter_bandpass_filter(prep, config.BP_LOW, config.BP_HIGH, config.FPS, order=3)
            except: pass
            
            if np.std(prep) > 1e-6:
                prep = (prep - np.mean(prep)) / np.std(prep)
            else:
                prep = prep - np.mean(prep)
                
            p_hr = en.hr_model.predict(prep.reshape(1, -1, 1), verbose=0)[0][0]
            p_spo2 = (0.7 * fe.calc_spo2(win)) + (0.3 * en.predict_spo2(win))
            
            time_ms = ((i + ws // 2) / fps) * 1000
            idx = (gt_data['time_ms'] - time_ms).abs().idxmin()
            closest = gt_data.loc[idx]
            
            p_hr_list.append(p_hr)
            p_spo2_list.append(p_spo2)
            gt_hr_list.append(closest['hr'])
            gt_spo2_list.append(closest['spo2'])
            
        if p_hr_list:
            mae_h = np.mean(np.abs(np.array(p_hr_list) - np.array(gt_hr_list)))
            mae_s = np.mean(np.abs(np.array(p_spo2_list) - np.array(gt_spo2_list)))
            print(f"  {subject_name} -> MAE HR: {mae_h:.2f}, MAE SpO2: {mae_s:.2f}")
            results.append({'subject': subject_name, 'version': version, 'mae_hr': mae_h, 'mae_spo2': mae_s})
            
    if results:
        df = pd.DataFrame(results)
        output_file = f"benchmark_universal_{'masked' if use_skin_mask else 'unmasked'}.csv"
        df.to_csv(output_file, index=False)
        return df
    return None

if __name__ == "__main__":
    # Add paths for both Dataset 1 and Dataset 2
    # The script will now look for subjects in these locations
    paths = [
        'Data/UBFC-rPPG/DATASET_1',
        'Data/UBFC-rPPG/DATASET_2',
        r'D:\project\datasets\archive' # Path to downloaded Dataset 2
    ]
    
    # Filter out paths that don't exist locally
    valid_paths = [p for p in paths if os.path.exists(p)]
    
    print("=== STARTING UNIVERSAL EXPERIMENTAL VALIDATION ===")
    print(f"Searching in: {valid_paths}")
    
    start_time = time.time()
    
    df_unmasked = benchmark(valid_paths, use_skin_mask=False)
    df_masked = benchmark(valid_paths, use_skin_mask=True)

    end_time = time.time()
    if df_unmasked is not None and df_masked is not None:
        print("\n=== FINAL UNIVERSAL COMPARISON ===")
        print(f"Baseline Avg MAE HR: {df_unmasked['mae_hr'].mean():.2f} BPM")
        print(f"Proposed Avg MAE HR: {df_masked['mae_hr'].mean():.2f} BPM")
        print(f"Baseline Avg SpO2 MAE: {df_unmasked['mae_spo2'].mean():.2f}%")
        print(f"Proposed Avg SpO2 MAE: {df_masked['mae_spo2'].mean():.2f}%")
        print(f"Total processing time: {(end_time - start_time)/60:.2f} minutes.")
