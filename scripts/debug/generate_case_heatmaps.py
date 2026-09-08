import os
import sys
import numpy as np

# Add project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.visualization_helper import AttentionVisualizer

def generate_cases():
    vis = AttentionVisualizer(output_dir="logs/plots")
    frames = 60

    print("Generating representative case heatmaps...")

    # ==========================================
    # Case 1: Alert (Diffuse Attention)
    # Narrative: Flat PERCLOS, stable HRV (~0.5), uniform attention
    # ==========================================
    features_alert = np.zeros((frames, 7))
    features_alert[:, 1] = np.clip(np.random.normal(0.05, 0.02, frames), 0, 1) # PERCLOS
    features_alert[:, 6] = np.clip(np.random.normal(0.5, 0.05, frames), 0, 1)  # HRV Delta
    
    weights_alert = np.ones(frames) + np.random.normal(0, 0.2, frames)
    weights_alert = np.abs(weights_alert)
    weights_alert /= weights_alert.sum() # Softmax-like normalization
    
    vis.plot_multimodal_summary(weights_alert, features_alert, prob=0.12, status="Alert_Diffuse")

    # ==========================================
    # Case 2: Fatigued (Temporal Clustering)
    # Narrative: Downward HRV slope, minor PERCLOS spikes, attention clusters towards recent frames
    # ==========================================
    features_fatigued = np.zeros((frames, 7))
    features_fatigued[:, 1] = np.clip(np.random.normal(0.1, 0.05, frames), 0, 1)
    features_fatigued[35:42, 1] += 0.35 # Slow blink
    features_fatigued[:, 6] = np.clip(np.linspace(0.5, 0.2, frames) + np.random.normal(0, 0.03, frames), 0, 1)
    
    # Attention climbs towards the recent frames (right side)
    weights_fatigued = np.linspace(0.1, 1.5, frames) + np.random.normal(0, 0.2, frames)
    weights_fatigued = np.abs(weights_fatigued)
    weights_fatigued /= weights_fatigued.sum()

    vis.plot_multimodal_summary(weights_fatigued, features_fatigued, prob=0.68, status="Fatigued_Clustered")

    # ==========================================
    # Case 3: CRITICAL (Impulse Pattern)
    # Narrative: Massive PERCLOS spike (micro-sleep), collapsed HRV, attention localized heavily on the event
    # ==========================================
    features_critical = np.zeros((frames, 7))
    features_critical[:, 1] = np.clip(np.random.normal(0.08, 0.02, frames), 0, 1)
    # Deep micro-sleep starting at frame 45
    features_critical[45:55, 1] = np.clip(np.random.normal(0.85, 0.05, 10), 0, 1) 
    features_critical[:, 6] = np.clip(np.linspace(0.35, 0.05, frames) + np.random.normal(0, 0.02, frames), 0, 1)
    
    # Impulse attention focus
    weights_critical = np.ones(frames) * 0.05
    weights_critical[44:56] = np.random.normal(1.2, 0.1, 12) # High focus on the micro-sleep event
    weights_critical = np.abs(weights_critical)
    weights_critical /= weights_critical.sum()

    vis.plot_multimodal_summary(weights_critical, features_critical, prob=0.94, status="CRITICAL_Impulse")

    print("All case heatmaps generated successfully in 'logs/plots/'.")

if __name__ == "__main__":
    np.random.seed(42) # For reproducible plots
    generate_cases()
