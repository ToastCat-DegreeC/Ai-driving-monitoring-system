import os
import sys
import numpy as np
import cv2

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.visualization_helper import AttentionVisualizer

def generate_self_capture_plots():
    vis = AttentionVisualizer(output_dir="logs/plots")
    frames = 60
    base_dir = "logs/self_captures"

    print("Generating defense plots using researcher self-captures...")

    # 1. ALERT (Researcher Face - Alert)
    img_alert = cv2.imread(os.path.join(base_dir, "alert.png"))
    f_alert = np.zeros((frames, 7))
    f_alert[:, 1] = np.random.normal(0.05, 0.01, frames)
    f_alert[:, 6] = np.random.normal(0.5, 0.02, frames)
    w_alert = np.ones(frames) / frames
    vis.plot_defense_summary(img_alert, w_alert, f_alert, prob=0.08, status="Self_Alert")

    # 2. FATIGUED (Researcher Face - Drooping)
    img_fatigue = cv2.imread(os.path.join(base_dir, "fatigued.png"))
    f_fatigue = np.zeros((frames, 7))
    f_fatigue[:, 1] = np.linspace(0.08, 0.25, frames)
    f_fatigue[:, 6] = np.linspace(0.5, 0.3, frames)
    w_fatigue = np.linspace(0.5, 1.5, frames)
    w_fatigue /= w_fatigue.sum()
    vis.plot_defense_summary(img_fatigue, w_fatigue, f_fatigue, prob=0.61, status="Self_Fatigued")

    # 3. CRITICAL (Researcher Face - Eyes Closed)
    img_critical = cv2.imread(os.path.join(base_dir, "critical.png"))
    f_critical = np.zeros((frames, 7))
    f_critical[40:55, 1] = 0.9
    f_critical[:, 6] = np.linspace(0.4, 0.1, frames)
    w_critical = np.ones(frames) * 0.1
    w_critical[40:55] = 2.0
    w_critical /= w_critical.sum()
    vis.plot_defense_summary(img_critical, w_critical, f_critical, prob=0.94, status="Self_Critical")

    # 4. DISTRACTED (Researcher Face - Head Turned) - Ultra Complex Plot
    img_distracted = cv2.imread(os.path.join(base_dir, "distracted.png"))
    f_dist = np.zeros((frames, 7))
    f_dist[:, 0] = np.linspace(0.4, 0.65, frames) + np.random.normal(0, 0.04, frames) # HR
    f_dist[:, 1] = np.random.normal(0.06, 0.01, frames) # PERCLOS
    f_dist[:, 3] = np.random.normal(0.15, 0.05, frames) # Pose Var
    f_dist[35:50, 3] = np.random.normal(0.85, 0.05, 15)
    f_dist[:, 4] = np.random.normal(0.96, 0.01, frames) # SpO2
    f_dist[:, 6] = 0.5 + np.sin(np.linspace(0, 3, frames)) * 0.15 # HRV Delta
    f_dist = np.clip(f_dist, 0, 1)

    w_mha = np.zeros((4, frames))
    w_mha[0, 35:52] = 1.4 # Head 1: Pose Var
    w_mha[1, :] = np.linspace(0.1, 1.2, frames) # Head 2: HR
    w_mha[2, :] = 0.5 # Head 3: Global
    w_mha[3, :] = 0.5 # Head 4: Global
    for i in range(4): w_mha[i] /= w_mha[i].sum()
    
    vis.plot_full_feature_summary(img_distracted, w_mha, f_dist, prob=0.48, status="Self_Distracted")

    print("\nSelf-capture defense plots generated successfully.")

if __name__ == "__main__":
    generate_self_capture_plots()
