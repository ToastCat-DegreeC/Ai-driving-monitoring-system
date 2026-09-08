import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from part_c.fusion_model import FusionModel
import os
import config

def main():
    # Load Empirical Data
    data_x_path = 'Data/nthu_pilot_X.npy'
    data_y_path = 'Data/nthu_pilot_Y.npy'
    
    if not os.path.exists(data_x_path) or not os.path.exists(data_y_path):
        print("Data files not found.")
        return

    X = np.load(data_x_path)
    Y = np.load(data_y_path)
    
    # Shuffle and Split (Same as training logic)
    np.random.seed(42) # Set seed for reproducibility
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    
    split = int(len(X) * 0.8)
    X_val, Y_val = X[split:], Y[split:]
    
    print(f"Evaluating on {len(X_val)} validation samples...")
    
    # Load Model
    weights_path = 'models/fusion_model_v3.h5'
    fusion = FusionModel(weights_path=weights_path)
    model = fusion.bilstm_model
    
    # Predict
    # Note: Model outputs [prob, weights]
    preds_prob, _ = model.predict(X_val, verbose=1)
    
    # Convert to binary labels
    # Using 0.5 as threshold for general classification metrics
    y_true = (Y_val > 0.5).astype(int)
    y_pred = (preds_prob > 0.5).astype(int)
    
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n=== FINAL VALIDATION METRICS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("================================")

if __name__ == "__main__":
    main()
