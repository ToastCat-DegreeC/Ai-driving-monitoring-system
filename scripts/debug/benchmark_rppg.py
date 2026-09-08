import cv2
import numpy as np
import os
import sys
import time
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
from utils.signal_processing import pad_or_truncate_signal
import config

def load_ground_truth(gt_path):
    """
    Loads HR and SpO2 from gtdump.xmp.
    Format: Timestep(ms), HR, SpO2, PPG
    """
    data = np.loadtxt(gt_path, delimiter=',')
    # Column 1: HR, Column 2: SpO2
    # The first column is Timestep, so indices are 1 and 2
    return data[:, 1], data[:, 2]

def benchmark_subject(subject_dir, feature_extractor, enhancement_net):
    video_path = os.path.join(subject_dir, "vid.avi")
    gt_path = os.path.join(subject_dir, "gtdump.xmp")
    
    if not os.path.exists(video_path) or not os.path.exists(gt_path):
        print(f"Skipping {subject_dir}: Missing video or ground truth.")
        return None

    print(f"Benchmarking {os.path.basename(subject_dir)}...")
    
    hr_gt, spo2_gt = load_ground_truth(gt_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    print(f"Video: {total_frames} frames @ {fps:.1f} FPS")

    # We need to map gt samples to frames. 
    # Usually, the gt sampling rate matches or is higher than video.
    # If len(gt) != total_frames, we might need interpolation.
    # Let's check.
    gt_len = len(hr_gt)
    
    predictions_hr = []
    ground_truths_hr = []
    predictions_spo2 = []
    ground_truths_spo2 = []

    rgb_means_buffer = []
    frame_count = 0
    
    # We process in batches of 30 frames (approx 1 sec) as in main.py
    BATCH_SIZE = 30
    WINDOW_SIZE = config.HR_MODEL_INPUT_LENGTH # 256
    
    # Store all RGB means to extract signal windows
    all_rgb_means = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Feature Extraction
        landmarks, forehead_roi, cheek_roi = feature_extractor.detect_and_get_landmarks(frame)
        rgb_mean = feature_extractor.get_roi_means(frame, landmarks, forehead_roi, cheek_roi)
        all_rgb_means.append(rgb_mean)
        
        frame_count += 1
        
        # 2. Every 30 frames, attempt a prediction if we have enough history
        if frame_count % BATCH_SIZE == 0 and len(all_rgb_means) >= WINDOW_SIZE:
            # Extract the last WINDOW_SIZE frames for prediction
            window_rgb_means = all_rgb_means[-WINDOW_SIZE:]
            
            # Predict HR and SpO2
            try:
                # Prepare signal for HR model
                raw_rppg_signal = feature_extractor.calc_rppg_signal(window_rgb_means)
                rppg_signal_prepared = pad_or_truncate_signal(raw_rppg_signal, config.HR_MODEL_INPUT_LENGTH)
                
                # Reshape for model (1, 256, 1)
                # Note: enhancement_net.enhance_features expects ear_signal too.
                # Since we don't have EAR ground truth here (or don't care for HR benchmark),
                # we can use dummy EAR signal or call models directly.
                
                # Let's call the models directly from enhancement_net to avoid EAR dependency
                rppg_input = rppg_signal_prepared.reshape(1, -1, 1)
                
                # Apply bandpass filter as in enhancement_network.py
                rppg_signal_filtered = enhancement_net._butter_bandpass_filter(rppg_signal_prepared, 0.7, 4.0, fps, order=3)
                if np.std(rppg_signal_filtered) > 1e-6:
                    rppg_signal_norm = (rppg_signal_filtered - np.mean(rppg_signal_filtered)) / np.std(rppg_signal_filtered)
                else:
                    rppg_signal_norm = rppg_signal_filtered - np.mean(rppg_signal_filtered)
                
                pred_hr = enhancement_net.hr_model.predict(rppg_signal_norm.reshape(1, 256, 1), verbose=0)[0][0]
                pred_spo2 = enhancement_net.predict_spo2(window_rgb_means)
                
                # Get Ground Truth for this frame
                # Map current frame index to GT index
                gt_idx = int((frame_count / total_frames) * (gt_len - 1))
                actual_hr = hr_gt[gt_idx]
                actual_spo2 = spo2_gt[gt_idx]
                
                if actual_hr > 0: # Some datasets have 0 for invalid data
                    predictions_hr.append(pred_hr)
                    ground_truths_hr.append(actual_hr)
                    
                if actual_spo2 > 0:
                    predictions_spo2.append(pred_spo2)
                    ground_truths_spo2.append(actual_spo2)

            except Exception as e:
                print(f"Error at frame {frame_count}: {e}")

        if frame_count >= 600:
            break

    cap.release()
    
    if not predictions_hr:
        return None
        
    mae_hr = np.mean(np.abs(np.array(predictions_hr) - np.array(ground_truths_hr)))
    mae_spo2 = np.mean(np.abs(np.array(predictions_spo2) - np.array(ground_truths_spo2)))
    
    print(f"  Result: HR MAE={mae_hr:.2f}, SpO2 MAE={mae_spo2:.2f}")
    return mae_hr, mae_spo2

def main():
    dataset_root = os.path.join(os.path.dirname(__file__), "../../Data/UBFC-rPPG/DATASET_1")
    subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    
    print(f"Found {len(subjects)} subjects in {dataset_root}")
    
    feature_extractor = FeatureExtractor()
    enhancement_net = EnhancementNetwork(
        weights_path_hr=os.path.join(os.path.dirname(__file__), "../../models/hr_model.h5"),
        weights_path_spo2=os.path.join(os.path.dirname(__file__), "../../models/spo2_model.h5")
    )
    
    results = []
    
    for sub in subjects:
        res = benchmark_subject(os.path.join(dataset_root, sub), feature_extractor, enhancement_net)
        if res:
            results.append(res)
    
    if results:
        results = np.array(results)
        avg_mae_hr = np.mean(results[:, 0])
        avg_mae_spo2 = np.mean(results[:, 1])
        print("\n" + "="*30)
        print("OVERALL BENCHMARK RESULTS")
        print(f"Average HR MAE:   {avg_mae_hr:.2f} BPM")
        print(f"Average SpO2 MAE: {avg_mae_spo2:.2f} %")
        print("="*30)
    else:
        print("No valid results collected.")

if __name__ == "__main__":
    main()
