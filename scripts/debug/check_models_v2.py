import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
from part_a.cnn_model import build_eye_cnn
from part_b.enhancement_network import EnhancementNetwork

def main():
    print("Checking model weights loading...")

    # 1. Check Eye CNN
    print("\n--- Checking Eye CNN ---")
    eye_cnn = build_eye_cnn()
    eye_weights = os.path.join(os.path.dirname(__file__), '../../models/eye_cnn_weights.h5')
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
            weights_path_hr=os.path.join(os.path.dirname(__file__), '../../models/hr_model.h5'),
            weights_path_perclos=os.path.join(os.path.dirname(__file__), '../../models/perclos_model.h5'),
            weights_path_spo2=os.path.join(os.path.dirname(__file__), '../../models/spo2_model.h5')
        )
        print("EnhancementNetwork initialization complete (check previous prints for weight loading details).")
    except Exception as e:
        print(f"Failed to initialize EnhancementNetwork: {e}")

    # 3. Check Live PERCLOS model
    print("\n--- Checking Live PERCLOS model ---")
    live_perclos_weights = os.path.join(os.path.dirname(__file__), '../../models/perclos_model_live.h5')
    if os.path.exists(live_perclos_weights):
        try:
            # Assuming it uses the same architecture as perclos_model
            perclos_model = enhancement._build_1d_cnn_perclos_model()
            perclos_model.load_weights(live_perclos_weights)
            print(f"Successfully loaded {live_perclos_weights}")
        except Exception as e:
            print(f"Failed to load {live_perclos_weights}: {e}")
    else:
        print(f"{live_perclos_weights} NOT found.")

if __name__ == "__main__":
    main()
