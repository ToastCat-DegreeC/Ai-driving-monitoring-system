import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np
import tensorflow as tf
from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
from scipy.signal import butter, filtfilt
import traceback
import config

# --- CONFIGURATION ---
HR_MODEL_WEIGHTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/hr_model.h5"))
RPPG_INPUT_LENGTH = config.HR_MODEL_INPUT_LENGTH
FPS = config.FPS  # Assumed FPS for the camera
BUFFER_SIZE = RPPG_INPUT_LENGTH # We need 256 frames for a full prediction

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_rppg(signal, hr=None, width=640, height=240):
    """
    Creates a plot of the rPPG signal on a numpy array.
    """
    plot_bg = np.zeros((height, width, 3), dtype=np.uint8)
    if signal is None or len(signal) < 2:
        cv2.putText(plot_bg, "Buffering signal...", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return plot_bg

    # Normalize signal for plotting
    plot_sig = signal - np.mean(signal)
    if np.std(plot_sig) > 1e-6:
        plot_sig = plot_sig / np.std(plot_sig)
    
    plot_sig = (plot_sig * (height / 6)) + (height / 2)
    plot_sig = plot_sig.astype(int)

    # Draw the signal
    points = np.array([[int(i * width / len(plot_sig)), plot_sig[i]] for i in range(len(plot_sig))])
    cv2.polylines(plot_bg, [points], isClosed=False, color=(0, 255, 0), thickness=2)
    
    if hr is not None:
        cv2.putText(plot_bg, f"Predicted HR: {hr:.1f} BPM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return plot_bg

def run_live_hr_test():
    """
    Performs a live test of HR estimation using the webcam.
    """
    print("Initializing components...")
    feature_extractor = FeatureExtractor(rppg_method='POS')
    enhancement_net = EnhancementNetwork(weights_path_hr=HR_MODEL_WEIGHTS)
    
    print("\nOpening camera...")
    cap = None
    for index in range(3):
        print(f"Trying camera index {index}...")
        temp_cap = cv2.VideoCapture(index)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret:
                cap = temp_cap
                print(f"Successfully opened camera {index}.")
                break
            else:
                temp_cap.release()
    
    if cap is None:
        print("Error: Could not open any camera.")
        return

    frames_buffer = []
    rppg_full_signal = []
    predicted_hr = None

    print("Starting live HR estimation... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            # 1. Feature Extraction (Extract 1 frame of rPPG)
            frames_buffer.append(frame)
            
            # Display feedback on frame
            cv2.putText(frame, f"Buffering: {len(rppg_full_signal)}/{BUFFER_SIZE}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if predicted_hr:
                cv2.putText(frame, f"HR: {predicted_hr:.1f} BPM", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # Draw face ROI
            landmarks, forehead_roi, cheek_roi = feature_extractor.detect_and_get_landmarks(frame)
            if forehead_roi:
                x, y, w, h = forehead_roi
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow('HR Monitoring System', frame)

            # 2. Extract rPPG every 10 frames to keep it smooth
            if len(frames_buffer) >= 10:
                batch_rppg = feature_extractor.extract_rppg(frames_buffer)
                if batch_rppg is not None:
                    rppg_full_signal.extend(batch_rppg)
                frames_buffer = []

            # 3. Predict HR when we have enough data
            if len(rppg_full_signal) >= BUFFER_SIZE:
                # Keep only the last BUFFER_SIZE samples
                input_signal = np.array(rppg_full_signal[-BUFFER_SIZE:])
                
                # Preprocessing: Filter
                try:
                    filtered_sig = butter_bandpass_filter(input_signal, 0.7, 4.0, FPS, order=3)
                    
                    # Preprocessing: Normalize (Standardization)
                    if np.std(filtered_sig) > 1e-6:
                        normalized_sig = (filtered_sig - np.mean(filtered_sig)) / np.std(filtered_sig)
                    else:
                        normalized_sig = filtered_sig - np.mean(filtered_sig)
                    
                    # Reshape for model (1, 256, 1)
                    model_input = normalized_sig.reshape(1, RPPG_INPUT_LENGTH, 1)
                    
                    # Predict
                    predicted_hr = enhancement_net.hr_model.predict(model_input, verbose=0)[0][0]
                    
                except Exception as e:
                    print(f"Prediction error: {e}")

                # Trim the full signal to avoid infinite growth
                rppg_full_signal = rppg_full_signal[-BUFFER_SIZE:]

            # 4. Plotting
            display_signal = np.array(rppg_full_signal) if rppg_full_signal else None
            plot_img = plot_rppg(display_signal, hr=predicted_hr)
            cv2.imshow('rPPG Signal', plot_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

    print("\nClosing application...")
    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_hr_test()
