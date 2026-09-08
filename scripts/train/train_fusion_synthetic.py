import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
from part_c.fusion_model import FusionModel
import config

# --- Configuration ---
NUM_SAMPLES = 5000
SEQ_LEN = config.FUSION_SEQUENCE_LEN
BATCH_SIZE = 32
EPOCHS = 30
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../../models/fusion_model.h5")

def generate_fusion_data(num_samples, seq_len):
    """
    Generates synthetic (HR, PERCLOS) sequences.
    HR is normalized to [0, 1] internally in FusionModel, 
    but for training, we'll generate raw and normalize here to match logic.
    """
    print(f"Generating {num_samples} synthetic fusion samples...")
    X = []
    Y = []
    
    for _ in range(num_samples):
        state = np.random.choice(['alert', 'fatigued', 'critical'])
        
        if state == 'alert':
            # Low PERCLOS, Stable/Normal HR
            hr = np.random.uniform(60, 80, seq_len)
            perclos = np.random.uniform(0.0, 0.05, seq_len)
            label = 0.1 + np.random.uniform(0, 0.2)
        elif state == 'fatigued':
            # Increasing PERCLOS, slightly varying HR
            hr = np.random.uniform(55, 90, seq_len)
            perclos = np.random.uniform(0.1, 0.25, seq_len)
            label = 0.5 + np.random.uniform(0, 0.2)
        else: # critical
            # High PERCLOS, potentially low or high HR
            hr = np.random.uniform(50, 110, seq_len)
            perclos = np.random.uniform(0.25, 0.5, seq_len)
            label = 0.8 + np.random.uniform(0, 0.2)
            
        # Combine and Normalize HR for the model
        seq = np.stack([hr, perclos], axis=1)
        seq[:, 0] = (seq[:, 0] - 40.0) / 120.0
        seq[:, 0] = np.clip(seq[:, 0], 0.0, 1.0)
        
        X.append(seq)
        Y.append(label)
        
    return np.array(X), np.array(Y)

def main():
    X, Y = generate_fusion_data(NUM_SAMPLES, SEQ_LEN)
    
    # Shuffle & Split
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    
    # Initialize and Train
    fusion = FusionModel()
    model = fusion.bilstm_model
    
    print("\n--- Training Fusion Model ---")
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
    )
    
    print(f"\nSaving weights to {SAVE_PATH}")
    model.save_weights(SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
