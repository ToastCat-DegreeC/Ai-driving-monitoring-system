# AI Driving Monitoring System (ADMS)
### Multi-Modal Vision & Contactless Physiological Sensing with Explainable AI

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-teal.svg)](https://mediapipe.dev/)
[![Status](https://img.shields.io/badge/Release-v3.1.0-brightgreen.svg)]()

---

## 📌 Overview

The **AI Driving Monitoring System (ADMS)** is a real-time, non-invasive driver safety solution developed through an industry-academia collaboration at **Tatung University (TTU)**. 

Traditional in-cabin Driver Monitoring Systems (DMS) rely almost exclusively on ocular metrics (PERCLOS, eye blink rate), making them vulnerable to sunglasses, head rotations, and late-stage detection. This project pioneers a **hybrid multimodal fusion framework** that integrates:
1. **Physical & Behavioral Tracking**: Real-time 3D head pose estimation, Eye Aspect Ratio (EAR/PERCLOS), and Mouth Aspect Ratio (MAR/yawning).
2. **Contactless Physiological Monitoring**: Remote Photoplethysmography (**rPPG**) extracted from facial micro-vascular color variations to estimate **Heart Rate (HR)**, Blood Oxygen Saturation (**SpO2**), and Heart Rate Variability (**HRV SDNN**).
3. **Multi-Head Attention (MHA) Temporal Fusion**: A 4-head attention mechanism over stacked **Bidirectional LSTMs (BiLSTM)** to dynamically assign importance weights across 7 physiological and behavioral features over a 2.0-second temporal window.
4. **Explainable AI (XAI)**: Live visualization of temporal attention weights, showing drivers and fleet safety operators *why* and *when* an anomaly was flagged.

---

## 🏛️ System Architecture

The architecture is structured into a decoupled, fail-operational three-tier pipeline:

```text
 ┌──────────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
 │   PART A: DISCOVERY  │ ──> │  PART B: REFINEMENT  │ ──> │    PART C: DECISION    │
 │ Facial & rPPG Engine │     │ Physiological 1D-CNNs│     │   MHA-BiLSTM Fusion    │
 └──────────────────────┘     └──────────────────────┘     └────────────────────────┘
  • MediaPipe (468 pts)        • 1D-CNN Heart Rate          • 2-Layer BiLSTM (64/32)
  • Dynamic ROI Switching      • 1D-CNN SpO2 (RoR)          • 4-Head Attention (d=16)
  • POS rPPG Projection        • 1D-CNN PERCLOS             • 7-Feature Context (60f)
  • 3D Head Pose (solvePnP)    • RR-Interval SDNN HRV       • Explainable AI (XAI)
```

### **Part A: Feature Extraction & Discovery** (`part_a/`)
* **Face Mesh & Landmark Tracking**: 468-point facial mesh tracking via MediaPipe.
* **SNR-Based Dynamic ROI Switching**: Evaluates the signal-to-noise ratio ($SNR_{dB}$) of the forehead, left cheek, and right cheek. If lighting or occlusion degrades one region, the pipeline dynamically pivots to the highest quality ROI.
* **rPPG Extraction**: Plane-Orthogonal-to-Skin (**POS**) projection separates pulsatile blood volume variations from non-rigid motion and illumination shifts.
* **3D Head Pose**: Perspective-n-Point (`solvePnP`) calculates Pitch, Yaw, and Roll Euler angles, tracking gaze deviation and head nodding.

### **Part B: Feature Refinement & Signal Processing** (`part_b/`)
* **1D-CNN Heart Rate Model**: Deep convolutional network processing filtered rPPG pulse waves (`models/hr_model.h5`).
* **1D-CNN SpO2 Model**: Hybrid neural model combining Ratio-of-Ratios ($AC/DC$) with learned feature representations (`models/spo2_model.h5`).
* **1D-CNN PERCLOS Model**: Analyzes continuous temporal sequences of eye openness (`models/perclos_model.h5`).
* **Biomedical RR-Interval Outlier Filter**: Robust peak detection with artifact rejection to compute true clinical-grade Heart Rate Variability (**SDNN**).

### **Part C: Temporal Fusion Engine (Decision Head)** (`part_c/`)
* Integrates a **7-dimensional temporal vector** over a 60-frame (2.0s) context window:
  $$\mathbf{X}_t = \left[ \text{HR}, \text{PERCLOS}, \text{MAR}, \sigma_{\text{pose}}, \text{SpO2}, \text{HRV}, \Delta\text{HRV} \right]$$
* **Multi-Head Attention (4 Heads, Key Dimension 16)**: Attends to different feature subspaces in parallel (e.g., Head 1 tracks eye blinks, Head 2 captures early physiological declines in HRV Delta).
* **Residual Transformer Connections**: Residual summation and Layer Normalization prevent gradient vanishing.
* **Regressive Output**: Predicts a continuous Fatigue Probability $P_{\text{fatigue}} \in [0, 1]$.

---

## 📊 Quantified Driver States

The system categorizes driver state into four operational levels:

| State | Fatigue Probability ($P$) | Trigger Criteria | Alert Level |
| :--- | :---: | :--- | :---: |
| **Alert** | $P \le 0.50$ | Normal blink rate, stable vitals, focused gaze | Normal (Green) |
| **Fatigued** | $0.50 < P \le 0.80$ | Droopy eyelids, frequent blinking, yawning, or HRV slump | Warning (Yellow) |
| **CRITICAL** | $P > 0.80$ | Extended eye closure ($\ge 1.5\text{s}$), micro-sleep, collapsed HRV | Emergency (Red) |
| **Distracted** | N/A | Head yaw $>20^\circ$, nod pitch $>20^\circ$, or gaze lost for $>5$ frames | Caution (Orange) |

---

## ⚡ Technical Benchmarks

Validated against the standard **UBFC-rPPG Dataset** (Dataset 1 & 2, $N=50$ subjects) and empirical driver datasets:

* **rPPG Heart Rate MAE**: **`13.52 BPM`** (UBFC-rPPG 50-subject benchmark)
* **rPPG SpO2 MAE**: **`8.21%`**
* **MHA-BiLSTM Fusion Validation MAE**: **`0.0332 – 0.0401`**
* **Real-Time Throughput**: Stable **`30 FPS`** on standard consumer CPU (Intel Core i7) via decoupled asynchronous background worker (`AIBackgroundWorker`).

---

## 🖥️ Live Telemetry Dashboard

The system features a professional 720p side-panel telemetry HUD rendered using OpenCV:

```text
 ┌───────────────────────────────────────┬───────────────────────────────┐
 │                                       │  DRIVER MONITOR               │
 │                                       ├───────────────────────────────┤
 │                                       │  HEART RATE:    74 BPM        │
 │                                       │  HRV (SDNN):    52.0 ms       │
 │                                       │  OXYGEN (SpO2): 98.2%         │
 │                                       ├───────────────────────────────┤
 │         LIVE WEBCAM STREAM            │  [ PULSE WAVE GRAPH ]         │
 │       (Face Mesh & 3D Pose)           ├───────────────────────────────┤
 │                                       │  FATIGUE PROBABILITY: 12%     │
 │                                       ├───────────────────────────────┤
 │                                       │  ATTENTION:     FOCUSED       │
 │                                       │  OCULAR STATE:  OPEN (12%)    │
 │                                       │  ALERTNESS:     ALERT         │
 │                                       ├───────────────────────────────┤
 │                                       │  TEMPORAL ATTENTION [MHA XAI] │
 │                                       │  -2.0s ─── [BARS] ─── Now     │
 └───────────────────────────────────────┴───────────────────────────────┘
```

* **Live Pulse Wave**: Real-time detrended ECG-style PPG waveform.
* **Ocular State Indicator**: Explicit `OPEN` (Green) vs. `CLOSED` (Red) PERCLOS monitoring.
* **Explainable AI (XAI) Widget**: Bar chart displaying temporal attention distribution over the past 2.0 seconds, dynamically highlighting in amber during blink or yawn events.

---

## 🚀 Quick Start

### 1. Prerequisites
* **OS**: Windows 10/11, Linux, or macOS
* **Python**: `3.9.x` (recommended for TensorFlow and MediaPipe compatibility)
* **Hardware**: Standard USB webcam or built-in laptop camera

### 2. Installation
Clone this repository and install the dependencies:

```bash
git clone https://github.com/ToastCat-DegreeC/Ai-driving-monitoring-system.git
cd Ai-driving-monitoring-system

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the System

```bash
# Run with default webcam (index 0)
python main.py

# Run with a specific camera index (e.g., USB camera 1)
python main.py --cam 1

# Run with a recorded test video file
python main.py --video path/to/video.mp4
```

* **Quit**: Press `q` while the video window is focused.
* **Resize**: The window can be freely resized or maximized (`cv2.WINDOW_NORMAL`).

---

## 📁 Repository Structure

```text
Ai-driving-monitoring-system/
├── config.py                 # Hyperparameters, paths, and clinical thresholds
├── main.py                   # Decoupled multi-threaded application controller
├── requirements.txt          # Python dependencies
├── version_patchnote.txt     # Complete version history and release patch notes
│
├── part_a/                   # Discovery & Feature Extraction
│   ├── feature_extractor.py  # MediaPipe landmarks, dynamic ROI switching, POS rPPG
│   └── cnn_model.py          # Eye patch CNN architecture
│
├── part_b/                   # Signal Refinement & Physiological Models
│   └── enhancement_network.py# 1D-CNN models for HR, SpO2, and PERCLOS
│
├── part_c/                   # Decision Engine & Explainable AI
│   └── fusion_model.py       # 4-Head Multi-Head Attention + BiLSTM model
│
├── models/                   # Pre-trained deep learning weights (.h5)
│   ├── fusion_model.h5       # MHA-BiLSTM Fusion Decision Engine
│   ├── hr_model.h5           # 1D-CNN Heart Rate Model
│   ├── perclos_model.h5      # 1D-CNN PERCLOS Model
│   ├── spo2_model.h5         # 1D-CNN SpO2 Model
│   └── eye_cnn_weights.h5    # Appearance-based Eye CNN weights
│
├── utils/                    # Helper modules
│   ├── gui_helper.py         # 720p side-panel HUD and XAI attention bar renderer
│   ├── signal_processing.py  # Filtering, POS math, and biomedical SDNN calculation
│   ├── calibration.py        # SystemCalibrator (60-frame zeroing routine)
│   └── logger.py             # Session CSV metric logging
│
└── scripts/                  # Training and validation suites
    ├── train/                # Training scripts for all sub-models
    │   ├── train_fusion_v3.py# MHA-BiLSTM training with synthetic/empirical data
    │   └── ...
    └── tests/                # Benchmarking and unit tests
        ├── benchmark_rppg.py # UBFC-rPPG automated benchmark script
        └── test_hrv_logic.py # Biomedical HRV unit test
```

---

## 👥 Authors & Academic Collaboration

* **Primary Researcher / Author**: Shan-Wei Chang (David Chen)
* **Institution**: Tatung University (TTU), Department of Computer Science & Engineering
* **Advising Professors**: 
  * Dr. Li Cho
  * Dr. Hsu Chao-Yun
* **Affiliation**: Academic-Industry Joint Research Initiative

---

## 📄 License

This project is licensed for academic research, education, and company demonstration purposes. See repository headers for detailed collaboration terms.
