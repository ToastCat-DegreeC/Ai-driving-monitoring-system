import cv2
import numpy as np
import os
import sys
import time
from collections import deque

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from part_a.feature_extractor import FeatureExtractor
from part_b.enhancement_network import EnhancementNetwork
from part_c.fusion_model import FusionModel
from utils.visualization_helper import AttentionVisualizer
import config

def process_video_real_extraction(video_source=0, output_name="Real_Extraction"):
    """
    Performs full real extraction (rPPG, HRV, PERCLOS) from a video source.
    """
    # 1. Initialize core system components
    extractor = FeatureExtractor()
    enhancer = EnhancementNetwork(
        weights_path_hr="models/hr_model.h5",
        weights_path_perclos="models/perclos_model.h5",
        weights_path_spo2="models/spo2_model.h5"
    )
    # The actual method in FusionModel is 'predict_fatigue'
    fusion = FusionModel(weights_path="models/fusion_model_v3.h5")
    vis = AttentionVisualizer(output_dir="logs/plots")
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    # Internal buffers required for physiological extraction
    sequence_len = config.FUSION_SEQUENCE_LEN # 60
    feature_buffer = deque(maxlen=sequence_len)
    rgb_means_buffer = deque(maxlen=256) # Part B needs a history for FFT/DSP
    
    print(f"Starting real extraction from {video_source}...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("End of video stream.")
                break
            
            # Part A: Landmark and ROI discovery
            landmarks, rois = extractor.detect_and_get_landmarks(frame)
            if landmarks is None: continue
            
            # Part A: Extract raw pixel means
            roi_means = extractor.get_roi_means(frame, landmarks, rois)
            rgb_means_buffer.append(roi_means)
            
            # Part A: Behavioral metrics
            ear = extractor.get_ear_value(frame, landmarks)
            mar = extractor.get_mar_value(frame, landmarks)
            pose, _ = extractor.get_head_pose(frame, landmarks)
            
            # Part B: Physiological refinement
            try:
                # We need enough frames for DSP (rPPG extraction)
                if len(rgb_means_buffer) < 64:
                    frame_count += 1
                    continue
                
                # 1. Extract the raw rPPG signal using ROI-Switching logic from Part A
                raw_rppg = extractor.calc_rppg_signal(list(rgb_means_buffer))
                
                # 2. Extract facial feature history for Part B's CNN (PERCLOS)
                # Part B needs a sequence of EAR values
                ear_history = [ear] * config.PERCLOS_MODEL_INPUT_LENGTH 
                
                # 3. Enhance features (HR, PERCLOS, HRV)
                hr, perclos, hrv = enhancer.enhance_features(raw_rppg, ear_history)
                
                # 4. Predict SpO2
                spo2 = enhancer.predict_spo2(list(rgb_means_buffer))
                
                # 5. Assemble raw features for Fusion (FusionModel.predict_fatigue handles normalization)
                # Note: HRV Delta is calculated internally by predict_fatigue using the buffer
                # Order: [HR, PERCLOS, MAR, Pose_Var, SpO2, HRV, HRV_Delta placeholder]
                raw_vector = [
                    hr, 
                    perclos, 
                    mar, 
                    np.mean(np.abs(pose)) if pose else 0,
                    spo2,
                    hrv,
                    0.0 # predict_fatigue will replace this with delta
                ]
                
                feature_buffer.append(raw_vector)
                frame_count += 1
            except Exception as e:
                print(f"Error in Part B Refinement: {e}")
                continue
            
            if frame_count % 20 == 0:
                print(f"Processed {frame_count} frames... Buffer: {len(feature_buffer)}/60")

            # Part C: Decision Fusion (requires 60 refined frames)
            if len(feature_buffer) == sequence_len:
                # The correct method name is 'predict_fatigue'
                status, prob, weights = fusion.predict_fatigue(list(feature_buffer))
                
                # package raw features for visualizer
                raw_features_array = np.array(feature_buffer)
                
                # In the raw_features_array, the 7th column is the HRV Delta placeholder
                # Let's populate it for visualization based on the buffer history
                for j in range(1, len(raw_features_array)):
                    delta = raw_features_array[j, 5] - raw_features_array[j-1, 5]
                    raw_features_array[j, 6] = 0.5 + delta / 10.0 # Scale for visibility
                
                # Normalize all features for visualization panel [0, 1]
                vis_features = raw_features_array.copy()
                vis_features[:, 0] = (vis_features[:, 0] - 60) / 40.0 # HR [60-100]
                vis_features[:, 4] = (vis_features[:, 4] - 90) / 10.0 # SpO2 [90-100]
                vis_features[:, 5] = (vis_features[:, 5]) / 50.0      # HRV
                vis_features = np.clip(vis_features, 0, 1)

                # Generate the 7-feature authentic plot
                save_path = vis.plot_authentic_7_feature_summary(
                    frame, 
                    weights.flatten(), 
                    vis_features, 
                    prob, 
                    f"{output_name}_{status}"
                )
                
                print(f"\nSUCCESS! 7-Feature Real plot generated: {save_path}")
                break 
                
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        print("Capture complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="0 for webcam or file path")
    parser.add_argument("--name", type=str, default="Authentic", help="Label for the plot")
    args = parser.parse_args()
    
    source = int(args.source) if args.source == "0" else args.source
    process_video_real_extraction(video_source=source, output_name=args.name)
