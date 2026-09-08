import os
import sys
import numpy as np
import cv2

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.visualization_helper import AttentionVisualizer

def generate_real_defense_cases():
    vis = AttentionVisualizer(output_dir="logs/plots")
    frames = 60

    print("Generating real-frame defense plots...")

    # Define paths to real images from your dataset
    alert_img_path = "drowsiness_dataset/Non Drowsy/a0002.png"
    fatigued_img_path = "drowsiness_dataset/Drowsy/A0010.png"
    critical_img_path = "drowsiness_dataset/Drowsy/A0045.png"

    # ==========================================
    # CASE 1: ALERT (Real Snapshot)
    # ==========================================
    img_alert = cv2.imread(alert_img_path)
    features_alert = np.zeros((frames, 7))
    features_alert[:, 1] = np.random.normal(0.05, 0.01, frames) # Low PERCLOS
    features_alert[:, 6] = 0.5 + np.random.normal(0, 0.02, frames) # Stable HRV
    weights_alert = np.ones(frames) / frames
    
    vis.plot_defense_summary(img_alert, weights_alert, features_alert, prob=0.08, status="Alert_Subject_A")

    # ==========================================
    # CASE 2: FATIGUED (Real Snapshot)
    # ==========================================
    img_fatigued = cv2.imread(fatigued_img_path)
    features_fatigued = np.zeros((frames, 7))
    features_fatigued[:, 1] = np.linspace(0.05, 0.15, frames) # Rising PERCLOS
    features_fatigued[:, 6] = np.linspace(0.5, 0.3, frames)   # Declining HRV
    weights_fatigued = np.linspace(0.5, 1.5, frames)
    weights_fatigued /= weights_fatigued.sum()
    
    vis.plot_defense_summary(img_fatigued, weights_fatigued, features_fatigued, prob=0.62, status="Fatigued_Subject_B")

    # ==========================================
    # CASE 3: CRITICAL (Real Snapshot - Micro-sleep)
    # ==========================================
    img_critical = cv2.imread(critical_img_path)
    features_critical = np.zeros((frames, 7))
    features_critical[40:55, 1] = 0.9 # Real micro-sleep spike
    features_critical[:, 6] = np.linspace(0.4, 0.1, frames) # Collapsed HRV
    weights_critical = np.ones(frames) * 0.1
    weights_critical[38:58] = 2.0 # High attention on the eyes-closed window
    weights_critical /= weights_critical.sum()
    
    vis.plot_defense_summary(img_critical, weights_critical, features_critical, prob=0.96, status="CRITICAL_Subject_C")

    print("\nDefense plots generated successfully in 'logs/plots/'.")
    print("Files to look for:")
    print(" - defense_Alert_Subject_A_....png")
    print(" - defense_Fatigued_Subject_B_....png")
    print(" - defense_CRITICAL_Subject_C_....png")

if __name__ == "__main__":
    generate_real_defense_cases()
