import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import cv2
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

class AttentionVisualizer:
    """
    Utility to visualize Multi-Head Attention weights for the Fusion Model.
    Generates heatmaps suitable for inclusion in the Thesis Draft.
    """
    def __init__(self, output_dir="logs/plots"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def plot_attention_heatmap(self, weights, feature_values=None, status="Normal", filename=None):
        """
        Plots a heatmap of attention weights across the 60-frame sequence.
        
        Args:
            weights: (60,) array of averaged attention weights
            feature_values: (60, 7) array of the input features (optional)
            status: Label for the current state (e.g., "Drowsy")
        """
        plt.figure(figsize=(12, 4))
        
        # Reshape for heatmap
        weights_reshaped = weights.reshape(1, -1)
        
        # Create heatmap
        ax = sns.heatmap(weights_reshaped, cmap="YlGnBu", cbar_kws={'label': 'Attention Weight'})
        
        plt.title(f"Temporal Attention Distribution (State: {status})")
        plt.xlabel("Look-back Frames (0 = Newest, 59 = Oldest)")
        plt.ylabel("Attention")
        
        # Invert x-axis so 0 is on the right (most recent)
        plt.gca().invert_xaxis()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attention_{status}_{timestamp}.png"
            
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Thesis-ready plot saved to: {save_path}")
        return save_path

    def plot_defense_summary(self, frame_img, weights, features, prob, status):
        """
        Triple-panel plot for thesis defense: Snapshot + Signals + Attention.
        """
        # Create a 3-panel layout - Adjusted height_ratios to make snapshot smaller
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.2, 0.4])
        
        ax0 = fig.add_subplot(gs[0]) # Snapshot (Smaller)
        ax1 = fig.add_subplot(gs[1]) # Signals (Larger)
        ax2 = fig.add_subplot(gs[2]) # Heatmap
        
        # 1. Plot Facial Snapshot
        if frame_img is not None:
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            ax0.imshow(img_rgb)
            ax0.set_title(f"Driver Snapshot (State: {status})", fontsize=14, fontweight='bold')
            ax0.axis('off')
        
        # 2. Plot Ocular vs Physiological signals (Apply light smoothing for professional look)
        frames = np.arange(60)
        perclos_smooth = gaussian_filter1d(features[:, 1], sigma=1.0)
        hrv_delta_smooth = gaussian_filter1d(features[:, 6], sigma=1.0)
        
        ax1.plot(frames, perclos_smooth, label="PERCLOS (Behavioral)", color='#1f77b4', linewidth=3)
        ax1.plot(frames, hrv_delta_smooth, label="HRV Delta (Physiological)", color='#d62728', linewidth=2, linestyle='--')
        ax1.set_title(f"Multimodal Signal Trends (Confidence: {prob:.2f})", fontsize=12)
        ax1.set_ylabel("Normalized Value [0, 1]")
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)
        
        # 3. Plot Attention Heatmap
        sns.heatmap(weights.reshape(1, -1), cmap="YlGnBu", ax=ax2, cbar=False)
        ax2.set_xlabel("Temporal Window (60 Frames / 2.0s Look-back)")
        ax2.set_ylabel("Attn")
        ax2.set_yticks([])
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_dir, f"defense_{status}_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Defense-ready plot saved to: {save_path}")
        return save_path

    def plot_authentic_7_feature_summary(self, frame_img, weights, features, prob, status):
        """
        Triple-panel plot showing all 7 REAL features and the averaged attention focus.
        """
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.2, 0.4])
        
        ax0 = fig.add_subplot(gs[0]) # Snapshot
        ax1 = fig.add_subplot(gs[1]) # All 7 Signals
        ax2 = fig.add_subplot(gs[2]) # Attention
        
        # 1. Plot Snapshot
        if frame_img is not None:
            img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            ax0.imshow(img_rgb)
            ax0.set_title(f"Authentic Driver Snapshot (State: {status})", fontsize=14, fontweight='bold')
            ax0.axis('off')
        
        # 2. Plot ALL 7 Features (Normalized for comparison)
        frames = np.arange(60)
        feature_names = ['HR', 'PERCLOS', 'MAR', 'Pose_Var', 'SpO2', 'HRV', 'HRV_Delta']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        # We normalize locally for the plot so lines are comparable [0, 1]
        for i in range(7):
            sig = features[:, i]
            # Simple min-max normalization for visual clarity if needed, 
            # but let's assume features are already in a reasonable range or pre-normalized
            sig_smooth = gaussian_filter1d(sig, sigma=1.0)
            ax1.plot(frames, sig_smooth, label=feature_names[i], color=colors[i], linewidth=2)
            
        ax1.set_title(f"Full Multimodal Data Stream (Confidence: {prob:.2f})", fontsize=12)
        ax1.set_ylabel("Value (Normalized Scale)")
        ax1.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))
        ax1.grid(alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)
        
        # 3. Plot Attention Heatmap
        sns.heatmap(weights.reshape(1, -1), cmap="YlGnBu", ax=ax2, cbar=False)
        ax2.set_xlabel("Temporal Window (60 Frames)")
        ax2.set_ylabel("Attn")
        ax2.set_yticks([])
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_dir, f"authentic_7f_{status}_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Authentic 7-feature plot saved to: {save_path}")
        return save_path




if __name__ == "__main__":
    # Test generation with dummy data
    vis = AttentionVisualizer()
    dummy_weights = np.random.dirichlet(np.ones(60), size=1).flatten()
    dummy_features = np.random.rand(60, 7)
    vis.plot_attention_heatmap(dummy_weights, status="Test_Run")
    print("Visualizer test successful.")
