import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
from part_c.fusion_model import FusionModel
import config

# --- Configuration ---
BATCH_SIZE = 32
EPOCHS = 50
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../../models/fusion_model.h5")
DATA_PATH_X = os.path.join(os.path.dirname(__file__), "../../data/nthu_pilot_X.npy")
DATA_PATH_Y = os.path.join(os.path.dirname(__file__), "../../data/nthu_pilot_Y.npy")

def normalize_fusion_data(X):
    """
    Applies the same normalization as FusionModel.predict_fatigue
    """
    X_norm = X.copy()
    # HR [40-160]
    X_norm[:, :, 0] = (X_norm[:, :, 0] - config.MIN_HR) / (config.MAX_HR - config.MIN_HR)
    # PERCLOS is already 0-1 (assumed)
    # MAR [0.0 - 1.0]
    X_norm[:, :, 2] = X_norm[:, :, 2] / config.MAR_MAX
    # Pose Variance [0 - 20]
    X_norm[:, :, 3] = X_norm[:, :, 3] / config.POSE_VAR_MAX
    # SpO2 [80 - 100]
    X_norm[:, :, 4] = (X_norm[:, :, 4] - config.SPO2_MIN) / (config.SPO2_MAX - config.SPO2_MIN)
    # HRV [0 - 150]
    X_norm[:, :, 5] = X_norm[:, :, 5] / config.HRV_MAX
    # HRV Delta [normalized to 0-1, centered at 0.5]
    X_norm[:, :, 6] = np.clip(X_norm[:, :, 6], -1, 1) * 0.5 + 0.5
    
    return np.clip(X_norm, 0, 1)

def generate_fusion_dataset(n_samples=8000, seq_len=config.FUSION_SEQUENCE_LEN):
    """
    Generates realistic 7-feature temporal sequences aligned with Chapter 3-4 of the thesis:
    Features: [HR, PERCLOS, MAR, Pose_Var, SpO2, HRV, HRV_Delta]
    """
    np.random.seed(42)
    X = np.zeros((n_samples, seq_len, 7), dtype=np.float32)
    Y = np.zeros(n_samples, dtype=np.float32)
    
    # Distribution: 40% Alert, 35% Fatigued, 25% Critical
    states = np.random.choice([0, 1, 2], size=n_samples, p=[0.40, 0.35, 0.25])
    
    for i in range(n_samples):
        st = states[i]
        
        if st == 0:
            # 1. ALERT STATE (0.05 - 0.20)
            hr = np.random.normal(72, 3.5, seq_len)
            # Low PERCLOS (0.0 - 0.12)
            perclos = np.clip(np.random.normal(0.04, 0.02, seq_len), 0.0, 0.15)
            # Normal MAR (0.08 - 0.22)
            mar = np.clip(np.random.normal(0.12, 0.03, seq_len), 0.05, 0.22)
            # Focused Pose (0.2 - 2.5 deg)
            pose_var = np.clip(np.random.normal(1.2, 0.4, seq_len), 0.1, 3.0)
            # Normal SpO2
            spo2 = np.clip(np.random.normal(98.5, 0.5, seq_len), 96.0, 100.0)
            # Normal HRV
            hrv = np.clip(np.random.normal(55, 5, seq_len), 40, 75)
            label = np.random.uniform(0.05, 0.20)
            
        elif st == 1:
            # 2. FATIGUED STATE (0.52 - 0.75)
            hr = np.random.normal(64, 4.5, seq_len)
            spo2 = np.clip(np.random.normal(97.0, 0.8, seq_len), 94.0, 99.0)
            hrv = np.clip(np.random.normal(35, 5, seq_len), 25, 50)
            
            variant = np.random.choice(['droop', 'yawn', 'hrv_drop'])
            if variant == 'droop':
                # Slow blinking or droop: PERCLOS 0.25 - 0.50
                target_p = np.random.uniform(0.28, 0.48)
                perclos = np.clip(np.linspace(0.08, target_p, seq_len) + np.random.normal(0, 0.03, seq_len), 0.15, 0.52)
                mar = np.clip(np.random.normal(0.12, 0.03, seq_len), 0.05, 0.25)
                pose_var = np.clip(np.random.normal(2.5, 1.0, seq_len), 0.5, 5.0)
                label = np.random.uniform(0.55, 0.72)
            elif variant == 'yawn':
                # Yawning episode
                perclos = np.clip(np.random.normal(0.15, 0.05, seq_len), 0.05, 0.30)
                yawn_peak = np.random.uniform(0.50, 0.75)
                mar = np.clip(np.sin(np.linspace(0, np.pi, seq_len)) * yawn_peak + np.random.normal(0.1, 0.02, seq_len), 0.08, 0.85)
                pose_var = np.clip(np.random.normal(2.2, 0.8, seq_len), 0.5, 4.5)
                label = np.random.uniform(0.58, 0.75)
            else:
                # HRV decline slump
                perclos = np.clip(np.random.normal(0.20, 0.04, seq_len), 0.10, 0.35)
                mar = np.clip(np.random.normal(0.12, 0.03, seq_len), 0.05, 0.25)
                pose_var = np.clip(np.random.normal(3.8, 1.2, seq_len), 1.0, 7.0)
                label = np.random.uniform(0.52, 0.68)
                
        else:
            # 3. CRITICAL STATE (0.84 - 0.98)
            hr = np.random.normal(58, 5, seq_len)
            spo2 = np.clip(np.random.normal(95.5, 1.0, seq_len), 91.0, 98.0)
            hrv = np.clip(np.random.normal(22, 4, seq_len), 15, 35)
            
            crit_variant = np.random.choice(['closed_eyes', 'nod_closure'])
            if crit_variant == 'closed_eyes':
                # Extended eye closure (PERCLOS 0.60 - 1.0)
                start_p = np.random.uniform(0.40, 0.60)
                end_p = np.random.uniform(0.85, 1.0)
                perclos = np.clip(np.linspace(start_p, end_p, seq_len) + np.random.normal(0, 0.02, seq_len), 0.50, 1.0)
                mar = np.clip(np.random.normal(0.12, 0.03, seq_len), 0.05, 0.25)
                pose_var = np.clip(np.random.normal(2.0, 0.8, seq_len), 0.5, 5.0)
            else:
                # Head nod + eye closure
                perclos = np.clip(np.random.uniform(0.55, 0.95, seq_len), 0.45, 1.0)
                mar = np.clip(np.random.normal(0.15, 0.04, seq_len), 0.05, 0.35)
                pose_var = np.clip(np.random.normal(9.0, 3.0, seq_len), 3.0, 18.0)
            label = np.random.uniform(0.84, 0.98)

        hrv_delta = np.zeros(seq_len, dtype=np.float32)
        hrv_delta[1:] = np.diff(hrv)

        X[i, :, 0] = hr
        X[i, :, 1] = perclos
        X[i, :, 2] = mar
        X[i, :, 3] = pose_var
        X[i, :, 4] = spo2
        X[i, :, 5] = hrv
        X[i, :, 6] = hrv_delta
        Y[i] = label

    return X, Y

def main():
    print("Generating comprehensive 7-feature training dataset...")
    X, Y = generate_fusion_dataset(n_samples=8000)
    X = normalize_fusion_data(X)
    print(f"Dataset Generated: X={X.shape}, Y={Y.shape}")
    
    # Shuffle & Split (80% Train, 20% Val)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    
    # Initialize Model
    fusion = FusionModel(weights_path=None) 
    model = fusion.bilstm_model
    
    print("\n--- Training MHA-BiLSTM Fusion Model (Interpretability Enabled) ---")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=['mse', None], 
        metrics={'fatigue_output': 'mae'}
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_fatigue_output_mae', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(SAVE_PATH, monitor='val_fatigue_output_mae', save_best_only=True, save_weights_only=True)
    ]
    
    history = model.fit(
        X_train, [Y_train, np.zeros((len(Y_train), config.FUSION_SEQUENCE_LEN))],
        validation_data=(X_val, [Y_val, np.zeros((len(Y_val), config.FUSION_SEQUENCE_LEN))]),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Also save to fusion_model_v3.h5
    save_path_v3 = os.path.join(os.path.dirname(__file__), "../../models/fusion_model_v3.h5")
    print(f"\nSaving best weights to {SAVE_PATH} and {save_path_v3}...")
    model.save_weights(SAVE_PATH)
    model.save(save_path_v3)
    
    val_mae = history.history['val_fatigue_output_mae'][-1]
    print(f"Training Completed. Best Validation MAE: {val_mae:.4f}")

if __name__ == "__main__":
    main()
