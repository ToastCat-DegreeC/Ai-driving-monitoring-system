import numpy as np
import os
import config
from part_c.fusion_model import FusionModel
from utils.visualization_helper import AttentionVisualizer

def generate_thesis_figure_4_1():
    print(">>> Generating Empirical Attention Heatmap (Figure 4.1)...")
    
    # 1. Load Data
    data_x_path = 'Data/nthu_pilot_X.npy'
    data_y_path = 'Data/nthu_pilot_Y.npy'
    
    if not os.path.exists(data_x_path):
        print("Error: Validation data not found.")
        return

    X = np.load(data_x_path)
    Y = np.load(data_y_path)
    
    # 2. Find a "Fatigue-Positive" sample (Y > 0.8)
    # We want a sample where the model is confident so the heatmap is high-signal
    indices = np.where(Y > 0.8)[0]
    if len(indices) == 0:
        sample_idx = 0
    else:
        sample_idx = indices[0]
        
    sample_x = X[sample_idx]
    sample_y = Y[sample_idx]
    
    # 3. Load Model and Predict Weights
    weights_path = 'models/fusion_model_v3.h5'
    fusion = FusionModel(weights_path=weights_path)
    
    # Predict (X expects batch dim)
    # Note: fusion_model.bilstm_model.predict returns [prob, weights]
    prob, weights = fusion.bilstm_model.predict(np.expand_dims(sample_x, axis=0), verbose=0)
    
    prob_val = float(prob[0][0])
    weight_vec = weights[0] # (60,)
    
    # 4. Use Visualizer to save professional plot
    # Save to the specific path referenced in the thesis .md
    os.makedirs('figures', exist_ok=True)
    vis = AttentionVisualizer(output_dir="figures")
    
    save_path = vis.plot_attention_heatmap(
        weights=weight_vec, 
        status="Drowsy", 
        filename="attention_heatmap_sample.png"
    )
    
    print(f"\nSUCCESS: Figure 4.1 generated from empirical sample #{sample_idx}")
    print(f"Sample Ground Truth: {sample_y:.4f}")
    print(f"Model Prediction:   {prob_val:.4f}")
    print(f"Saved to:           {save_path}")

if __name__ == "__main__":
    generate_thesis_figure_4_1()
