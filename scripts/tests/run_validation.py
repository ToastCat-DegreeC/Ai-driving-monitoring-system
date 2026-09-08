import cv2
import numpy as np
import os
import pandas as pd
import time
from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
import config
from utils.signal_processing import pad_or_truncate_signal

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

def benchmark(dataset_path, subjects, use_skin_mask=True):
    fe = FeatureExtractor()
    en = EnhancementNetwork(config.MODEL_PATH_HR, config.MODEL_PATH_PERCLOS, config.MODEL_PATH_SPO2)
    results = []
    
    print(f"\n>>> Starting Benchmark (Skin Masking: {use_skin_mask}) on {len(subjects)} subjects...")
    
    for subject in subjects:
        subject_dir = os.path.join(dataset_path, subject)
        video_path = os.path.join(subject_dir, 'vid.avi')
        gt_path = os.path.join(subject_dir, 'gtdump.xmp')
        
        if not os.path.exists(video_path) or not os.path.exists(gt_path):
            continue
            
        gt_data = load_gt_v1(gt_path)
        if gt_data is None: continue
        
        print(f"Processing {subject}...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        rgb_means_buffer = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            # Sub-sample to speed up (process every 2nd frame if needed, but benchmark usually processes all)
            l, fr, cr = fe.detect_and_get_landmarks(frame)
            if use_skin_mask:
                rgb_means_buffer.append(fe.get_roi_means(frame, l, fr, cr))
            else:
                roi = fr if fr else cr
                if roi:
                    roi = fe._clip_roi(roi, frame.shape)
                    if roi:
                        x, y, w, h = roi
                        roi_frame = frame[y:y+h, x:x+w]
                        rgb_means_buffer.append(np.mean(roi_frame, axis=(0, 1)) if roi_frame.size > 0 else np.zeros(3))
                    else: rgb_means_buffer.append(np.zeros(3))
                else: rgb_means_buffer.append(np.zeros(3))
                
            if frame_count % 300 == 0:
                print(f"  Frame {frame_count} processed...")
                
        cap.release()
        
        ws, step = config.RPPG_WINDOW_SIZE, 30
        p_hr_list, p_spo2_list, gt_hr_list, gt_spo2_list = [], [], [], []
        
        print(f"  Calculating metrics for {len(rgb_means_buffer)} frames...")
        for i in range(0, len(rgb_means_buffer) - ws, step):
            win = rgb_means_buffer[i : i + ws]
            sig = fe.calc_rppg_signal(win)
            prep = pad_or_truncate_signal(sig, config.HR_MODEL_INPUT_LENGTH)
            
            # Apply same filtering as in main.py
            try:
                prep = en._butter_bandpass_filter(prep, config.BP_LOW, config.BP_HIGH, config.FPS, order=3)
            except: pass
            
            if np.std(prep) > 1e-6:
                prep = (prep - np.mean(prep)) / np.std(prep)
            else:
                prep = prep - np.mean(prep)
                
            p_hr = en.hr_model.predict(prep.reshape(1, -1, 1), verbose=0)[0][0]
            
            # SpO2 Estimation (Hybrid)
            spo2_ai = en.predict_spo2(win)
            spo2_calc = fe.calc_spo2(win)
            p_spo2 = (0.7 * spo2_calc) + (0.3 * spo2_ai)
            
            time_ms = ((i + ws // 2) / fps) * 1000
            # Find closest GT
            idx = (gt_data['time_ms'] - time_ms).abs().idxmin()
            closest = gt_data.loc[idx]
            
            p_hr_list.append(p_hr)
            p_spo2_list.append(p_spo2)
            gt_hr_list.append(closest['hr'])
            gt_spo2_list.append(closest['spo2'])
            
        if p_hr_list:
            mae_h = np.mean(np.abs(np.array(p_hr_list) - np.array(gt_hr_list)))
            mae_s = np.mean(np.abs(np.array(p_spo2_list) - np.array(gt_spo2_list)))
            print(f"  {subject} -> MAE HR: {mae_h:.2f}, MAE SpO2: {mae_s:.2f}")
            results.append({'subject': subject, 'mae_hr': mae_h, 'mae_spo2': mae_s})
            
    if results:
        df = pd.DataFrame(results)
        output_file = f"benchmark_local_{'masked' if use_skin_mask else 'unmasked'}.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        return df
    return None

if __name__ == "__main__":
    dataset_path = 'Data/UBFC-rPPG/DATASET_1'
    # Detect all directories ending in -gt
    subjects = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d.endswith('-gt')]
    subjects.sort() # Ensure consistent order
    
    print(f"Detected subjects: {subjects}")
    print("=== STARTING FULL LOCAL EXPERIMENTAL VALIDATION ===")
    start_time = time.time()
    
    df_unmasked = benchmark(dataset_path, subjects, use_skin_mask=False)
    df_masked = benchmark(dataset_path, subjects, use_skin_mask=True)

    end_time = time.time()
    print(f"\nValidation completed in {(end_time - start_time)/60:.2f} minutes.")

    if df_unmasked is not None and df_masked is not None:
        print("\n=== VALIDATION SUMMARY FOR THESIS ===")
        avg_mae_unmasked = df_unmasked['mae_hr'].mean()
        avg_mae_masked = df_masked['mae_hr'].mean()
        print(f"Baseline (Unmasked) Avg MAE HR: {avg_mae_unmasked:.2f} BPM")
        print(f"Proposed (Skin-Masked) Avg MAE HR: {avg_mae_masked:.2f} BPM")
        
        improvement = ((avg_mae_unmasked - avg_mae_masked) / avg_mae_unmasked) * 100
        print(f"Net Accuracy Improvement: {improvement:.2f}%")
        
        # SpO2 summary
        avg_spo2_mae_unmasked = df_unmasked['mae_spo2'].mean()
        avg_spo2_mae_masked = df_masked['mae_spo2'].mean()
        print(f"Baseline SpO2 MAE: {avg_spo2_mae_unmasked:.2f}%")
        print(f"Proposed SpO2 MAE: {avg_spo2_mae_masked:.2f}%")
