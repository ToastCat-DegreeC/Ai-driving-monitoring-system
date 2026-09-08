import cv2
import numpy as np
import time
import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
from utils.signal_processing import pad_or_truncate_signal

# --- Configuration ---
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../../models/perclos_model_live.h5")
SEQ_LENGTH = 128  # Must match EnhancementNetwork.ear_input_length
RECORD_DURATION = 10  # Seconds per recording
BATCH_SIZE = 8
EPOCHS = 10

def record_ear_sequence(feature_extractor, duration=10):
    """
    Records EAR values from the webcam for a specified duration.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print(f"Recording for {duration} seconds... Please look at the camera.")
    
    ear_values = []
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks, _, _ = feature_extractor.detect_and_get_landmarks(frame)
        ear = feature_extractor.get_ear_value(frame, landmarks)
        ear_values.append(ear)
        
        # Display the frame with EAR value
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {duration - (time.time() - start_time):.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Recording PERCLOS Data', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if len(ear_values) == 0:
        return None
        
    # Standardize to SEQ_LENGTH
    return pad_or_truncate_signal(ear_values, SEQ_LENGTH)

def main():
    feature_extractor = FeatureExtractor()
    enhancement_net = EnhancementNetwork()
    model = enhancement_net.perclos_model
    
    # Load existing weights if they exist
    base_model_path = os.path.join(os.path.dirname(__file__), "../../models/perclos_model.h5")
    if os.path.exists(base_model_path):
        try:
            model.load_weights(base_model_path)
            print("Loaded existing PERCLOS weights.")
        except Exception as e:
            print(f"Warning: Could not load existing weights: {e}")

    X = []
    Y = []
    
    EAR_THRESHOLD = 0.2 # Threshold to calculate ground truth PERCLOS from recorded signal

    while True:
        print("\n--- New Data Point ---")
        label_input = input("Enter label (0 for Alert, 1 for Drowsy, 's' to stop and train, 'q' to quit): ").lower()
        
        if label_input == 'q':
            return
        if label_input == 's':
            break
            
        try:
            # We use the label to guide the user's behavior, 
            # but PERCLOS is a continuous value.
            # We will calculate the 'Ground Truth' PERCLOS from the recorded EAR signal 
            # based on the frames where EAR < threshold.
            
            if label_input == '0':
                print("Behavior: Alert (Blink normally)")
            elif label_input == '1':
                print("Behavior: Drowsy (Frequent blinking or slow closures)")
            else:
                print("Invalid input. Try again.")
                continue
                
            ear_seq = record_ear_sequence(feature_extractor, duration=RECORD_DURATION)
            
            if ear_seq is not None:
                # Calculate ground truth PERCLOS for this specific recorded sequence
                closed_frames = np.sum(np.array(ear_seq) < EAR_THRESHOLD)
                perclos_label = closed_frames / SEQ_LENGTH
                
                print(f"Captured sequence. Calculated PERCLOS: {perclos_label:.4f}")
                
                X.append(np.array(ear_seq).reshape(SEQ_LENGTH, 1))
                Y.append(perclos_label)
            else:
                print("Failed to capture sequence.")
                
        except ValueError:
            print("Invalid input.")

    if len(X) == 0:
        print("No data recorded. Exiting.")
        return

    X = np.array(X)
    Y = np.array(Y)

    print(f"\nTraining on {len(X)} recorded samples...")
    
    # Fine-tune the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print(f"Saving fine-tuned model to {SAVE_PATH}")
    model.save_weights(SAVE_PATH)
    
    # Ask if user wants to overwrite the main model
    overwrite = input(f"Overwrite 'perclos_model.h5' with new weights? (y/n): ").lower()
    if overwrite == 'y':
        model.save_weights(base_model_path)
        print("Main model updated.")

if __name__ == "__main__":
    main()
