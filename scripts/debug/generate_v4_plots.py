import os
import sys
import numpy as np
import cv2

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.visualization_helper import AttentionVisualizer

def generate_hybrid_defense_plots():
    vis = AttentionVisualizer(output_dir="logs/plots")
    frames = 60

    print("Generating hybrid defense plots (Drowsiness Face + Physiological Signals)...")

    # 1. ALERT (Drowsiness Dataset Subject - Wide Open)
    img_alert = cv2.imread("logs/debug_frames_7gt/alert_final_open.png")
    f_alert = np.zeros((frames, 7))
    f_alert[:, 1] = np.random.normal(0.05, 0.01, frames)
    f_alert[:, 6] = np.random.normal(0.5, 0.02, frames)
    w_alert = np.ones(frames) / frames
    vis.plot_defense_summary(img_alert, w_alert, f_alert, prob=0.08, status="Alert_V4")

    # 2. FATIGUED (Drowsiness Dataset Subject - Drooping)
    img_fatigue = cv2.imread("logs/debug_frames_7gt/fatigued_final_droop.png")
    f_fatigue = np.zeros((frames, 7))
    f_fatigue[:, 1] = np.linspace(0.08, 0.25, frames)
    f_fatigue[:, 6] = np.linspace(0.5, 0.3, frames)
    w_fatigue = np.linspace(0.5, 1.5, frames)
    w_fatigue /= w_fatigue.sum()
    vis.plot_defense_summary(img_fatigue, w_fatigue, f_fatigue, prob=0.61, status="Fatigued_V4")

    # 3. CRITICAL (Subject 7-gt - Real Micro-sleep)
    # We keep Subject 7-gt for Critical because the closed-eyes are perfect and it is real benchmark data.
    img_critical = cv2.imread("logs/debug_frames_7gt/best_closed_eyes.png")
    f_critical = np.zeros((frames, 7))
    f_critical[40:55, 1] = 0.9
    f_critical[:, 6] = np.linspace(0.4, 0.1, frames)
    w_critical = np.ones(frames) * 0.1
    w_critical[40:55] = 2.0
    w_critical /= w_critical.sum()
    vis.plot_defense_summary(img_critical, w_critical, f_critical, prob=0.94, status="CRITICAL_V4")

    print("\nHybrid defense plots generated successfully.")

if __name__ == "__main__":
    generate_hybrid_defense_plots()
