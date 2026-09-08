import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

print(sys.path)
try:
    import cv2
    print("cv2 imported")
    import numpy
    print("numpy imported")
    import mediapipe
    print("mediapipe imported")
    import tensorflow
    print("tensorflow imported")
    from part_a.feature_extractor import FeatureExtractor
    print("FeatureExtractor imported")
except Exception as e:
    print(f"Error: {e}")
