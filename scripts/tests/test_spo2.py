import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from part_a.feature_extractor import FeatureExtractor

def test_spo2():
    print("Initializing FeatureExtractor...")
    fe = FeatureExtractor()
    
    print("Generating synthetic RGB means data (256 frames)...")
    t = np.linspace(0, 10, 256)
    
    # Simulate BGR channels (B=0, G=1, R=2)
    # Add a DC offset and a small AC sine wave (simulating pulse)
    r = 120.0 + 3.0 * np.sin(2 * np.pi * 1.2 * t) 
    g = 100.0 + 2.0 * np.sin(2 * np.pi * 1.2 * t)
    b = 90.0 + 1.5 * np.sin(2 * np.pi * 1.2 * t)
    
    rgb_means = np.column_stack((b, g, r))
    
    print("Calculating SpO2...")
    spo2 = fe.calc_spo2(list(rgb_means))
    print(f"-> Calculated SpO2: {spo2:.2f}%")
    
    if 80.0 <= spo2 <= 100.0:
        print("Test passed: SpO2 is within realistic bounds.")
    else:
        print("Test failed: SpO2 is out of bounds.")

if __name__ == '__main__':
    test_spo2()