import numpy as np
import tensorflow as tf
import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from part_b.enhancement_network import EnhancementNetwork

# --- Configuration ---
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../../models/perclos_model.h5")
NUM_SAMPLES = 5000
SEQ_LENGTH = 128 # Must match EnhancementNetwork.ear_input_length
BATCH_SIZE = 32
EPOCHS = 20

def generate_synthetic_data(num_samples, seq_length):
    """
    Generates synthetic EAR sequences and their corresponding PERCLOS labels.
    """
    print(f"Generating {num_samples} synthetic samples...")
    
    X = [] # Input: EAR sequences
    Y = [] # Label: PERCLOS value (0.0 to 1.0) 
    
    # Threshold for considering an eye "closed" (standard research value is often 80% closure, or EAR < 0.15-0.2)
    EAR_THRESHOLD = 0.2
    
    for _ in range(num_samples):
        # Decide if this sample represents an "Alert" or "Drowsy" state
        # to ensure we have a good distribution of data.
        state_type = np.random.choice(['alert', 'drowsy', 'semidrowsy'])
        
        sequence = np.ones(seq_length) * 0.3 # Base open eye EAR ~0.3
        
        if state_type == 'alert':
            # Alert: 0-2 quick blinks
            num_blinks = np.random.randint(0, 3)
            for _ in range(num_blinks):
                blink_len = np.random.randint(2, 5) # Fast blink (2-5 frames)
                start = np.random.randint(0, seq_length - blink_len)
                sequence[start:start+blink_len] = np.random.uniform(0.05, 0.15, blink_len)
                
            # Add small noise
            noise = np.random.normal(0, 0.01, seq_length)
            sequence += noise

        elif state_type == 'drowsy':
            # Drowsy: Long blinks (micro-sleeps) or drooping
            num_microsleeps = np.random.randint(1, 4)
            for _ in range(num_microsleeps):
                sleep_len = np.random.randint(15, 40) # Long blink (0.5s - 1.3s)
                start = np.random.randint(0, seq_length - sleep_len) if seq_length > sleep_len else 0
                sequence[start:start+sleep_len] = np.random.uniform(0.02, 0.12, min(sleep_len, seq_length))
            
            # Or general droopiness
            if np.random.rand() > 0.5:
                sequence = sequence * np.random.uniform(0.5, 0.8) # Lower baseline

            # Add noise
            noise = np.random.normal(0, 0.01, seq_length)
            sequence += noise

        elif state_type == 'semidrowsy':
             # Frequent medium blinks
            num_blinks = np.random.randint(3, 6)
            for _ in range(num_blinks):
                blink_len = np.random.randint(5, 12) 
                start = np.random.randint(0, seq_length - blink_len)
                sequence[start:start+blink_len] = np.random.uniform(0.05, 0.18, blink_len)
            noise = np.random.normal(0, 0.01, seq_length)
            sequence += noise

        # Clip values to realistic range
        sequence = np.clip(sequence, 0.0, 0.4)
        
        # Calculate Ground Truth PERCLOS for this sequence
        # PERCLOS = (Number of frames where eye is closed) / (Total frames)
        closed_frames = np.sum(sequence < EAR_THRESHOLD)
        perclos_label = closed_frames / seq_length
        
        X.append(sequence.reshape(seq_length, 1))
        Y.append(perclos_label)
        
    return np.array(X), np.array(Y)

def main():
    # 1. Generate Data
    X, Y = generate_synthetic_data(NUM_SAMPLES, SEQ_LENGTH)
    
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
    # We only need the perclos model part
    enhancement_net = EnhancementNetwork() 
    model = enhancement_net.perclos_model
    
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
    print("Done. You can now run main.py!")

if __name__ == "__main__":
    main()
