import numpy as np
import tensorflow as tf
import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from part_b.enhancement_network import EnhancementNetwork

# --- Configuration ---
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../../models/spo2_model.h5")
NUM_SAMPLES = 5000
SEQ_LENGTH = 256 # Must match EnhancementNetwork.rppg_input_length
BATCH_SIZE = 32
EPOCHS = 20

def generate_synthetic_spo2_data(num_samples, seq_length):
    """
    Generates synthetic RGB sequences and their corresponding SpO2 labels.
    """
    print(f"Generating {num_samples} synthetic SpO2 samples...")
    X = []
    Y = []
    
    for _ in range(num_samples):
        # Time array
        t = np.linspace(0, 10, seq_length)
        
        # Randomize heart rate between 50 and 120 bpm (0.8 to 2.0 Hz)
        hr_hz = np.random.uniform(0.8, 2.0)
        
        # Decide if this sample is healthy (95-100) or low (85-94)
        is_healthy = np.random.rand() > 0.3
        
        if is_healthy:
            target_spo2 = np.random.uniform(95.0, 100.0)
            # Normal amplitude ratio (Red signal usually cleaner/stronger)
            ac_r_amp = np.random.uniform(1.0, 3.0)
            ac_b_amp = ac_r_amp * np.random.uniform(0.4, 0.7) 
        else:
            target_spo2 = np.random.uniform(85.0, 94.0)
            # Altered amplitude ratio to simulate lower SpO2
            ac_r_amp = np.random.uniform(1.0, 3.0)
            ac_b_amp = ac_r_amp * np.random.uniform(0.8, 1.2) 
            
        # DC components (baseline color intensities)
        dc_r = np.random.uniform(100.0, 150.0)
        dc_g = np.random.uniform(80.0, 120.0)
        dc_b = np.random.uniform(70.0, 110.0)
        
        # Generate raw channels with noise
        r = dc_r + ac_r_amp * np.sin(2 * np.pi * hr_hz * t) + np.random.normal(0, 0.5, seq_length)
        g = dc_g + (ac_r_amp * 0.8) * np.sin(2 * np.pi * hr_hz * t) + np.random.normal(0, 0.5, seq_length)
        b = dc_b + ac_b_amp * np.sin(2 * np.pi * hr_hz * t) + np.random.normal(0, 0.5, seq_length)
        
        # Combine into BGR
        rgb_means = np.column_stack((b, g, r))
        
        # Normalize it the same way predict_spo2 does inside EnhancementNetwork
        rgb_means_processed = np.zeros_like(rgb_means)
        for i in range(3):
            std = np.std(rgb_means[:, i])
            if std > 1e-6:
                rgb_means_processed[:, i] = (rgb_means[:, i] - np.mean(rgb_means[:, i])) / std
            else:
                rgb_means_processed[:, i] = rgb_means[:, i] - np.mean(rgb_means[:, i])
                
        X.append(rgb_means_processed)
        Y.append(target_spo2)
        
    return np.array(X), np.array(Y)

def main():
    # 1. Generate Data
    X, Y = generate_synthetic_spo2_data(NUM_SAMPLES, SEQ_LENGTH)

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")

    # 2. Initialize Model
    enhancement_net = EnhancementNetwork()
    model = enhancement_net.spo2_model

    # 3. Train
    print("\n--- Starting Training ---")
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
    )

    # 4. Save
    print(f"\n--- Saving weights to {SAVE_PATH} ---")
    model.save_weights(SAVE_PATH)
    print("Done. You can now update main.py to use the AI model for SpO2!")

if __name__ == "__main__":
    main()
