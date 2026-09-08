import cv2
import numpy as np
import os
import pandas as pd
from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
import config
from utils.signal_processing import pad_or_truncate_signal

def load_gt_v1(gt_path):
    """
    Loads ground truth from gtdump.xmp (Dataset 1).
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
    Line 1: PPG, Line 2: HR, Line 3: Time(s)
    """
    try:
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        ppg = np.fromstring(lines[0], sep=' ')
        hr = np.fromstring(lines[1], sep=' ')
        time_s = np.fromstring(lines[2], sep=' ')
        return pd.DataFrame({'time_ms': time_s * 1000, 'hr': hr, 'spo2': 98.0, 'ppg': ppg})
    except Exception as e:
        print(f"Error loading GT V2 {gt_path}: {e}")
        return None

def benchmark(limit=None, use_skin_mask=True):
    dataset_path = r'D:\project\datasets\archive'
    subjects = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('subject')]
    if not subjects:
        subjects = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d.endswith('-gt')]
    
    if limit: subjects = subjects[:limit]
    
    fe = FeatureExtractor()
    en = EnhancementNetwork(config.MODEL_PATH_HR, config.MODEL_PATH_PERCLOS, config.MODEL_PATH_SPO2)
    results = []
    
    print(f"\n>>> Starting Benchmark (Skin Masking: {use_skin_mask}) on {len(subjects)} subjects...")
    
    for subject in subjects:
        subject_dir = os.path.join(dataset_path, subject)
        video_path = os.path.join(subject_dir, 'vid.avi')
        gt_v1, gt_v2 = os.path.join(subject_dir, 'gtdump.xmp'), os.path.join(subject_dir, 'ground_truth.txt')
        
        if not os.path.exists(video_path): continue
        gt_data = load_gt_v2(gt_v2) if os.path.exists(gt_v2) else load_gt_v1(gt_v1)
        if gt_data is None: continue
        
        print(f"Processing {subject}...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        rgb_means_buffer = []
        while True:
            ret, frame = cap.read()
            if not ret: break
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
        cap.release()
        
        ws, step = config.RPPG_WINDOW_SIZE, 30
        p_hr_list, p_spo2_list, gt_hr_list, gt_spo2_list = [], [], [], []
        
        for i in range(0, len(rgb_means_buffer) - ws, step):
            win = rgb_means_buffer[i : i + ws]
            sig = fe.calc_rppg_signal(win)
            prep = pad_or_truncate_signal(sig, config.HR_MODEL_INPUT_LENGTH)
            try: prep = en._butter_bandpass_filter(prep, 0.7, 4.0, config.FPS, order=3)
            except: pass
            
            if np.std(prep) > 1e-6: prep = (prep - np.mean(prep)) / np.std(prep)
            else: prep = prep - np.mean(prep)
                
            p_hr = en.hr_model.predict(prep.reshape(1, -1, 1), verbose=0)[0][0]
            p_spo2 = (0.7 * fe.calc_spo2(win)) + (0.3 * en.predict_spo2(win))
            
            time_ms = ((i + ws // 2) / fps) * 1000
            closest = gt_data.iloc[(gt_data['time_ms'] - time_ms).abs().argsort()[:1]]
            
            p_hr_list.append(p_hr)
            p_spo2_list.append(p_spo2)
            gt_hr_list.append(closest['hr'].values[0])
            gt_spo2_list.append(closest['spo2'].values[0])
            
        if p_hr_list:
            mae_h = np.mean(np.abs(np.array(p_hr_list) - np.array(gt_hr_list)))
            mae_s = np.mean(np.abs(np.array(p_spo2_list) - np.array(gt_spo2_list)))
            print(f"  {subject} -> MAE HR: {mae_h:.2f}, MAE SpO2: {mae_s:.2f}")
            results.append({'subject': subject, 'mae_hr': mae_h, 'mae_spo2': mae_s})
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(f"benchmark_results_{'masked' if use_skin_mask else 'unmasked'}.csv", index=False)
        return df
    return None

if __name__ == "__main__":
    print("=== STARTING FULL THESIS EXPERIMENT ===")
    df_unmasked = benchmark(limit=None, use_skin_mask=False)
    df_masked = benchmark(limit=None, use_skin_mask=True)

    if df_unmasked is not None and df_masked is not None:
        print("\n=== FINAL THESIS COMPARISON ===")
        print(f"Baseline (Unmasked) MAE HR: {df_unmasked['mae_hr'].mean():.2f} BPM")
        print(f"Proposed (Skin-Masked) MAE HR: {df_masked['mae_hr'].mean():.2f} BPM")
        imp = ((df_unmasked['mae_hr'].mean() - df_masked['mae_hr'].mean()) / df_unmasked['mae_hr'].mean()) * 100
        print(f"Net Improvement: {imp:.2f}%")
