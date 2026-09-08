import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
from part_a.cnn_model import build_eye_cnn
from part_b.enhancement_network import EnhancementNetwork
import config

def get_root_path(rel_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', rel_path))

def main():
    print("Checking model weights loading with config.py...")

    # 1. Check Eye CNN
    print("\n--- Checking Eye CNN ---")
    eye_cnn = build_eye_cnn()
    eye_weights = get_root_path(config.MODEL_PATH_EYE_CNN)
    if os.path.exists(eye_weights):
        try:
            eye_cnn.load_weights(eye_weights)
            print(f"Successfully loaded {eye_weights}")
        except Exception as e:
            print(f"Failed to load {eye_weights}: {e}")
    else:
        print(f"{eye_weights} NOT found.")

    # 2. Check Enhancement Network
    print("\n--- Checking Enhancement Network Models ---")
    try:
        enhancement = EnhancementNetwork(
            weights_path_hr=get_root_path(config.MODEL_PATH_HR),
            weights_path_perclos=get_root_path(config.MODEL_PATH_PERCLOS),
            weights_path_spo2=get_root_path(config.MODEL_PATH_SPO2)
        )
        print("EnhancementNetwork initialization complete.")
    except Exception as e:
        print(f"Failed to initialize EnhancementNetwork: {e}")

if __name__ == "__main__":
    main()
