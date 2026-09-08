import os
import sys
import numpy as np
import cv2

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.visualization_helper import AttentionVisualizer

def generate_complex_7feature_case():
    vis = AttentionVisualizer(output_dir="logs/plots")
    frames = 60

    print("Generating complex 7-feature case study (Distraction + Restlessness)...")

    # 1. Load a snapshot that shows distraction (looking away)
    # zc1707.png in the Non Drowsy set shows a subject with head turned
    img_distracted = cv2.imread("drowsiness_dataset/Non Drowsy/zc1707.png")
    
    # 2. Simulate ALL 7 Features
    # Order: [HR, PERCLOS, MAR, Pose_Var, SpO2, HRV, HRV_Delta]
    f = np.zeros((frames, 7))
    
    # HR: Rising due to restlessness/anxiety
    f[:, 0] = np.linspace(0.4, 0.65, frames) + np.random.normal(0, 0.04, frames)
    # PERCLOS: Low (driver is awake but distracted)
    f[:, 1] = np.random.normal(0.06, 0.01, frames)
    # MAR: Stable (closed mouth)
    f[:, 2] = np.random.normal(0.05, 0.02, frames)
    # Pose Variance: HIGH (Sharp head rotation at frame 40)
    f[:, 3] = np.random.normal(0.15, 0.05, frames)
    f[35:50, 3] = np.random.normal(0.85, 0.05, 15) # Sudden distraction event
    # SpO2: Stable
    f[:, 4] = np.random.normal(0.96, 0.01, frames)
    # HRV: Erratic
    f[:, 5] = np.random.normal(0.4, 0.05, frames)
    # HRV Delta: Fluctuating
    f[:, 6] = 0.5 + np.sin(np.linspace(0, 3, frames)) * 0.15
    
    f = np.clip(f, 0, 1)

    # 3. Simulate Multi-Head Attention (4 Heads specializing in different features)
    w_mha = np.zeros((4, frames))
    
    # Head 1: Focuses on the Distraction event (Pose_Var) at frame 40
    w_mha[0, :] = 0.05
    w_mha[0, 35:52] = 1.4
    
    # Head 2: Focuses on the rising Heart Rate (Restlessness)
    w_mha[1, :] = np.linspace(0.1, 1.2, frames)
    
    # Head 3: Focuses on HRV Delta oscillations
    w_mha[2, :] = 0.5 + np.random.normal(0, 0.1, frames)
    
    # Head 4: Global monitoring (Diffuse)
    w_mha[3, :] = 0.5
    
    # Normalize each head
    for i in range(4):
        w_mha[i] /= w_mha[i].sum()
    
    vis.plot_full_feature_summary(img_distracted, w_mha, f, prob=0.48, status="Distracted_Restless")

    print("\nComplex 7-feature plot generated in 'logs/plots/'.")

if __name__ == "__main__":
    generate_complex_7feature_case()
