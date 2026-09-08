import cv2

# --- Model Paths ---
MODEL_PATH_HR = 'models/hr_model.h5'
MODEL_PATH_PERCLOS = 'models/perclos_model.h5'
MODEL_PATH_SPO2 = 'models/spo2_model.h5'
MODEL_PATH_FUSION = 'models/fusion_model.h5'
MODEL_PATH_EYE_CNN = 'models/eye_cnn_weights.h5'

# --- System Settings ---
THREADED_MODE = True
DEBUG_VISUALS = True
SQI_THRESHOLD = -10.0 # Effectively disabled for testing
SNR_MIN_THRESH = -8.0
SPO2_SMOOTHING = 0.05 # Very slow smoothing to prevent drops

# --- Signal Processing Constants ---
FPS = 30
HR_MODEL_INPUT_LENGTH = 256
PERCLOS_MODEL_INPUT_LENGTH = 128
RPPG_WINDOW_SIZE = HR_MODEL_INPUT_LENGTH
RPPG_BUFFER_SIZE = RPPG_WINDOW_SIZE
EAR_WINDOW_SIZE = PERCLOS_MODEL_INPUT_LENGTH
SAMPLE_INTERVAL = 1.0 / FPS
BP_LOW = 0.7
BP_HIGH = 4.0

# --- Heart Rate Stabilization ---
HR_ALPHA = 0.15
MAX_HR_JUMP = 12.0
MIN_HR = 45.0
MAX_HR = 160.0
HRV_MAX = 150.0
HR_HISTORY_LEN = 10
HR_STABLE_INITIAL = 70.0

# --- Eye & Drowsiness Thresholds ---
YAWN_THRESH = 0.5
YAWN_FRAMES = 20
EAR_CLOSED_THRESH = 0.22
EYE_OPEN_PROB_THRESH = 0.35
EYE_SMOOTHING_FACTOR = 0.3

# --- Distraction Thresholds ---
DISTRACTION_THRESH_PITCH = 20
DISTRACTION_THRESH_YAW = 20
CALIBRATION_FRAMES = 60
HEAD_POSE_SMOOTHING = 0.25 # Lower = smoother, less jitter

# --- SpO2 Constants ---
SPO2_A = 123.0
SPO2_B = 25.0
SPO2_MIN = 80.0
SPO2_MAX = 100.0

# --- Fusion Model ---
FUSION_SEQUENCE_LEN = 60
MHA_HEADS = 4
MHA_KEY_DIM = 16
MAR_MAX = 0.6
POSE_VAR_MAX = 20.0
FATIGUE_THRESHOLD_FATIGUED = 0.5
FATIGUE_THRESHOLD_CRITICAL = 0.8

# --- MediaPipe Settings ---
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# --- UI & Dashboard Settings ---
DASHBOARD_WIDTH = 340
GRAPH_LEN = 100
COLOR_BG = (30, 30, 30)
COLOR_TEXT = (240, 240, 240)
COLOR_ACCENT = (0, 255, 0)
COLOR_ALERT = (0, 0, 255)
COLOR_GRAPH = (255, 200, 0)

# --- Landmarks ---
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
MOUTH_LANDMARKS = [13, 14, 78, 308]

# --- Head Pose Landmarks ---
# 1: Nose, 152: Chin, 33: R-Eye Outer, 263: L-Eye Outer, 61: R-Mouth, 291: L-Mouth
POSE_LANDMARKS = [1, 152, 263, 33, 291, 61] 
# Note: In our 3D model: 263(L) -> X-, 33(R) -> X+, 291(L) -> X-, 61(R) -> X+
