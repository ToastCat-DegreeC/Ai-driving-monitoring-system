import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    import mediapipe
    print("mediapipe imported successfully")
except Exception as e:
    print(f"Error importing mediapipe: {e}")
