import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))

models = [
    'hr_model.h5',
    'perclos_model.h5',
    'spo2_model.h5',
    'eye_cnn_weights.h5',
    'perclos_model_live.h5'
]

for model_name in models:
    model_path = os.path.join(models_dir, model_name)
    if os.path.exists(model_path):
        try:
            # For weights file, we can't load it directly as a model
            if 'weights' in model_name:
                print(f"File {model_name} exists at {model_path}.")
            else:
                model = tf.keras.models.load_model(model_path)
                print(f"Successfully loaded {model_name} from {model_path}")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    else:
        print(f"Model file {model_name} NOT found at {model_path}.")
