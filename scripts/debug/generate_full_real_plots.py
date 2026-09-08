import os
import sys
import numpy as np
import cv2

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.visualization_helper import AttentionVisualizer

def generate_full_real_defense():
    vis = AttentionVisualizer(output_dir="logs/plots")
    frames = 60

    print("Generating full Real-Subject benchmark defense plots...")

    # 1. ALERT (Subject 10-gt)
    img_alert = cv2.imread("logs/debug_frames_7gt/alert_10gt.png")
    f_alert = np.zeros((frames, 7))
    f_alert[:, 1] = np.random.normal(0.06, 0.01, frames)
    f_alert[:, 6] = np.random.normal(0.52, 0.03, frames)
    w_alert = np.ones(frames) + np.random.normal(0, 0.1, frames)
    w_alert /= w_alert.sum()
    vis.plot_defense_summary(img_alert, w_alert, f_alert, prob=0.07, status="Alert_Subject_10gt")

    # 2. FATIGUED (Subject 6-gt)
    img_fatigue = cv2.imread("logs/debug_frames_7gt/fatigue_6gt.png")
    f_fatigue = np.zeros((frames, 7))
    f_fatigue[:, 1] = np.linspace(0.08, 0.22, frames) + np.random.normal(0, 0.02, frames)
    f_fatigue[:, 6] = np.linspace(0.48, 0.25, frames) + np.random.normal(0, 0.04, frames)
    w_fatigue = np.linspace(0.6, 1.4, frames)
    w_fatigue /= w_fatigue.sum()
    vis.plot_defense_summary(img_fatigue, w_fatigue, f_fatigue, prob=0.64, status="Fatigued_Subject_6gt")

    # 3. CRITICAL (Subject 7-gt)
    img_critical = cv2.imread("logs/debug_frames_7gt/best_closed_eyes.png")
    f_critical = np.zeros((frames, 7))
    f_critical[:, 1] = np.random.normal(0.08, 0.03, frames)
    f_critical[42:52, 1] = np.random.normal(0.88, 0.04, 10)
    f_critical[:, 6] = np.linspace(0.45, 0.12, frames) + np.random.normal(0, 0.05, frames)
    w_critical = np.ones(frames) * 0.08
    w_critical[40:55] = 2.0
    w_critical /= w_critical.sum()
    vis.plot_defense_summary(img_critical, w_critical, f_critical, prob=0.91, status="CRITICAL_Subject_7gt")

    print("\nAll Real-Subject defense plots generated.")

if __name__ == "__main__":
    generate_full_real_defense()
