import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import csv
import numpy as np
import tensorflow as tf
import cv2
from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
from utils.signal_processing import pad_or_truncate_signal
from scipy.signal import butter, filtfilt
import gc

# --- CONFIGURATION ---
UBFC_DATASET_ROOT = r"D:\project\datasets\archive"
HR_MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "../../models/hr_model.h5")

# Training Hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
RPPG_INPUT_LENGTH = 256
FPS = 30
WINDOW_SIZE_FRAMES = RPPG_INPUT_LENGTH 

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def load_ubfc_data_generator(dataset_root, fe):
    if not os.path.exists(dataset_root):
        print(f"Error: Path {dataset_root} not found.")
        return

    subjects = [s for s in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, s)) and s.startswith('subject')]
    if not subjects: subjects = [s for s in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, s)) and s.endswith('-gt')]
    
    print(f"Found {len(subjects)} subjects.")

    for s_folder in subjects:
        s_path = os.path.join(dataset_root, s_folder)
        video_p = os.path.join(s_path, "vid.avi")
        gt_v1, gt_v2 = os.path.join(s_path, "gtdump.xmp"), os.path.join(s_path, "ground_truth.txt")

        if not os.path.exists(video_p): continue
        print(f"Loading {s_folder}...")

        try:
            if os.path.exists(gt_v2):
                with open(gt_v2, 'r') as f: lines = f.readlines()
                hr_sig = np.fromstring(lines[1], sep=' ')
            elif os.path.exists(gt_v1):
                hr_sig = []
                with open(gt_v1, 'r') as f:
                    r = csv.reader(f)
                    for row in r: 
                        if len(row) >= 2: hr_sig.append(float(row[1]))
                hr_sig = np.array(hr_sig)
            else: continue
        except: continue
        
        cap = cv2.VideoCapture(video_p)
        rgb_m = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            l, fr, cr = fe.detect_and_get_landmarks(frame)
            rgb_m.append(fe.get_roi_means(frame, l, fr, cr))
        cap.release()

        if not rgb_m: continue
        r_sig = fe.calc_rppg_signal(rgb_m)
        del rgb_m
        gc.collect()

        try: f_sig = butter_bandpass_filter(r_sig, 0.7, 4.0, FPS)
        except: f_sig = r_sig

        mlen = min(len(f_sig), len(hr_sig))
        f_sig, hr_sig = f_sig[:mlen], hr_sig[:mlen]
        
        step = int(WINDOW_SIZE_FRAMES * 0.2) # 80% overlap
        for i in range(0, mlen - WINDOW_SIZE_FRAMES, step):
            win_r = f_sig[i : i + WINDOW_SIZE_FRAMES]
            m_hr = np.mean(hr_sig[i : i + WINDOW_SIZE_FRAMES])
            if np.std(win_r) > 1e-6: win_r = (win_r - np.mean(win_r)) / np.std(win_r)
            else: win_r = win_r - np.mean(win_r)
            yield win_r.reshape(WINDOW_SIZE_FRAMES, 1), m_hr

def main():
    fe = FeatureExtractor()
    X, Y = [], []
    for x, y in load_ubfc_data_generator(UBFC_DATASET_ROOT, fe):
        X.append(x); Y.append(y)
    
    if not X: return
    X, Y = np.array(X), np.array(Y)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]
    
    split = int(len(X) * 0.8)
    X_t, X_v, Y_t, Y_v = X[:split], X[split:], Y[:split], Y[split:]
    
    model = EnhancementNetwork().hr_model
    lr_s = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    e_s = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    print("\n--- Starting Retraining ---")
    model.fit(X_t, Y_t, validation_data=(X_v, Y_v), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[lr_s, e_s])
    model.save_weights(HR_MODEL_WEIGHTS_PATH)
    print(f"Saved optimized weights to {HR_MODEL_WEIGHTS_PATH}")

if __name__ == "__main__": main()
