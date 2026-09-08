import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np
from part_a.feature_extractor import FeatureExtractor
import time

def plot_rppg(signal, width=640, height=240):
    """
    Creates a plot of the rPPG signal on a numpy array.
    """
    plot_bg = np.zeros((height, width, 3), dtype=np.uint8)
    if signal is None or len(signal) < 2:
        cv2.putText(plot_bg, "Waiting for signal...", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return plot_bg

    # Normalize signal to fit in the plot height
    signal = signal - np.mean(signal)
    if np.std(signal) > 0:
        signal = signal / np.std(signal)
    
    signal = (signal * (height / 4)) + (height / 2)
    signal = signal.astype(int)

    # Draw the signal
    points = np.array([[int(i * width / len(signal)), signal[i]] for i in range(len(signal))])
    cv2.polylines(plot_bg, [points], isClosed=False, color=(0, 255, 0), thickness=2)
    
    return plot_bg

def run_live_rppg_test():
    """
    Performs a live test of the rPPG extraction using the laptop camera
    and plots the signal in a separate window.
    """
    print("Initializing FeatureExtractor...")
    feature_extractor = FeatureExtractor()
    print("FeatureExtractor initialized.")

    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frames_buffer = []
    FRAME_BUFFER_SIZE = 50 # Process 50 frames at a time
    rppg_signal = None

    print("Starting live capture... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Add frame to buffer
        frames_buffer.append(frame)

        # --- Face Detection for Visualization ---
        landmarks, forehead_roi, cheek_roi = feature_extractor.detect_and_get_landmarks(frame)
        if forehead_roi:
            x, y, w, h = forehead_roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display buffer status on the frame
        buffer_text = f"Buffer: {len(frames_buffer)}/{FRAME_BUFFER_SIZE}"
        cv2.putText(frame, buffer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Live Camera Feed', frame)

        # --- Process Buffer and Update Plot ---
        if len(frames_buffer) >= FRAME_BUFFER_SIZE:
            print(f"\nProcessing {FRAME_BUFFER_SIZE} frames for rPPG extraction...")
            
            start_time = time.time()
            rppg_signal = feature_extractor.extract_rppg(frames_buffer)
            end_time = time.time()
            
            print(f"Processing time: {end_time - start_time:.2f} seconds")

            if rppg_signal is not None and rppg_signal.ndim == 1:
                print("rPPG signal extracted successfully.")
            else:
                print("Could not extract a valid rPPG signal from the buffer.")
                rppg_signal = None

            # Clear the buffer for the next batch
            frames_buffer = []

        # --- Create and Display Plot in a Separate Window ---
        plot_img = plot_rppg(rppg_signal, width=640, height=480)
        cv2.imshow('rPPG Signal Plot', plot_img)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print("\nClosing application...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_live_rppg_test()
