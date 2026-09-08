# AI Driving Monitoring System: System Overview

This document provides a high-level explanation of the system architecture and the functional roles of its core components. The system is designed to detect driver fatigue and distraction in real-time using a standard RGB camera.

---

## 1. System Architecture
The project is divided into three primary functional layers (**Part A, B, and C**) coordinated by a central multi-threaded controller (`main.py`).

### Data Flow:
`Camera Input` → `Part A (Features)` → `Part B (Enhancement)` → `Part C (Fusion)` → `Dashboard UI`

---

## 2. Core Components

### 🟢 main.py (The Controller)
The "brain" of the application. It handles:
- **Multi-threading**: Runs an `AIBackgroundWorker` thread so that heavy model inference doesn't lag the camera display.
- **Input Management**: Manages webcam/USB camera selection and video file playback.
- **UI Rendering**: Combines the camera feed with a real-time dashboard showing HR, SpO2, and Fatigue status.
- **Calibration**: Performs an initial 60-frame baseline calibration for Head Pose and Eye Aspect Ratio (EAR).

### 🔵 Part A: feature_extractor.py (Signal Discovery)
This module uses **MediaPipe Face Mesh** to extract raw data from the video frames.
- **Landmark Detection**: Tracks 468 3D facial landmarks.
- **ROI Extraction**: Automatically identifies the **Forehead** and **Cheeks** to extract skin pixel data.
- **rPPG Pre-processing**: Calculates the mean RGB values from skin regions to be used for heart rate detection.
- **Ocular/Oral Analysis**: Calculates the **EAR (Eye Aspect Ratio)** for blink detection and **MAR (Mouth Aspect Ratio)** for yawning detection.
- **Head Pose**: Estimates Pitch, Yaw, and Roll to detect if the driver is looking away from the road.

### 🟡 Part B: enhancement_network.py (Feature Refinement)
This layer uses Deep Learning (1D-CNNs) to turn raw signals into physiological metrics.
- **Heart Rate (HR) Model**: A 1D-CNN that filters the raw rPPG (color change) signal to predict the heart rate.
- **PERCLOS Model**: Analyzes the Eye Aspect Ratio (EAR) sequence to calculate the "Percentage of Eye Closure" (a primary indicator of drowsiness).
- **SpO2 Model**: A specialized network that analyzes the ratio of Red vs. Blue color changes to estimate blood oxygen levels.
- **Signal Filtering**: Implements Butterworth Bandpass filters to remove noise caused by lighting changes or small movements.

### 🔴 Part C: fusion_model.py (The Decision Engine)
The final layer that performs **Multi-modal Fusion**.
- **BiLSTM Model**: A Bidirectional Long Short-Term Memory (LSTM) network that looks at the *history* of your HR and PERCLOS over time.
- **Temporal Analysis**: Instead of just looking at a single second, it analyzes trends (e.g., "Is the heart rate dropping while eye closure is increasing?").
- **Fatigue Classification**: Outputs a final probability and status:
    - **Alert**: Normal driving state.
    - **Drowsy**: Early signs of fatigue detected.
    - **CRITICAL**: High risk of falling asleep; immediate intervention suggested.

---

## 3. Utility Modules
- **config.py**: Centralized configuration for all thresholds (YAWN_THRESH, EAR_CLOSED_THRESH), model paths, and UI colors.
- **utils/signal_processing.py**: Helper functions for signal padding, truncation, and normalization.
- **utils/gui_helper.py**: Handles the drawing of the dashboard, graphs, and alert text onto the OpenCV window.

---

## 4. Execution
The system can be run with various parameters:
- `python main.py` (Default webcam)
- `python main.py --cam 2` (Specific USB camera)
- `python main.py --video path/to/video.mp4` (Video file analysis)

---

## 5. Prerequisites & Installation

### Hardware Requirements
- **Camera**: Standard RGB Webcam or USB camera (720p at 30fps recommended).
- **CPU**: Intel Core i5 or equivalent (required for real-time inference).
- **GPU**: Optional (if available, the system will use it via TensorFlow).

### Software Requirements
- **Python**: 3.8 to 3.10 (Recommended: 3.9.x)
- **Pip**: Latest version

### Installation Steps
1.  **Clone the Repository**:
    ```powershell
    git clone <repository_url>
    cd "AI driving monitoring system"
    ```
2.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

### Core Packages Used:
- **`opencv-python`**: For camera capture, image processing, and UI dashboard.
- **`tensorflow`**: Powering the HR, SpO2, and Fusion BiLSTM models.
- **`mediapipe`**: For high-performance facial landmark tracking.
- **`numpy`**: For numerical operations on skin pixel means and EAR/MAR signals.
- **`scipy`**: For Butterworth bandpass filtering.
- **`scikit-learn`**: Used in training and some signal evaluation scripts.
- **`matplotlib`**: (Optional) Used for plotting graphs during debugging or training.
