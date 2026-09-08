---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')

# **AI Driving Monitoring System**
### Technical Update: Calibration & Fusion Optimization
**Date:** March 13, 2026
**Role:** Senior Software Engineer

---

# **Overview: Today's Milestones**

1. **Robust Calibration**: Subject-Specific EAR Normalization.
2. **Fusion Optimization**: BiLSTM Regression for Fatigue.
3. **Weight Persistence**: Fusion Model weight-loading mechanism.
4. **Validation**: Batch Testing across 7 subjects (UBFC-rPPG).
5. **Architectural Flexibility**: Argparse for offline/online testing.

---

# **1. Robust Calibration**
### **Subject-Specific EAR Normalization**

- **The Problem**: Lighting and physiological differences caused false "Drowsy" alerts (PERCLOS > 0.90) for alert subjects.
- **The Solution**: Implemented a **60-frame startup calibration routine**.
- **Result**:
    - Calculates a baseline "Open Eye" EAR for each unique user.
    - Normalizes real-time EAR to a standard **0.0 - 0.4 range**.
    - Resolved major bias in PERCLOS detection logic.

---

# **2. Fusion Model Optimization**
### **BiLSTM Regression for Fatigue**

- **Input Normalization**:
    - Implemented Min-Max scaling for Heart Rate (40-160 BPM range).
    - Prevents HR from "drowning out" subtle PERCLOS signals.
- **Improved Loss Function**:
    - Re-trained Fusion BiLSTM using **Mean Squared Error (MSE)**.
    - Resulted in significantly more responsive fatigue probability outputs.

---

# **3. System Integration**
### **Weight Persistence & Flexibility**

- **Model Loading**:
    - Updated `FusionModel` to support persistent weight loading (`fusion_model.h5`).
    - Ensures training progress is preserved across different sessions.
- **Command-Line Flexibility**:
    - Added `argparse` to `main.py` for `--video` file input.
    - Facilitates robust offline testing in diverse environments.

---

# **4. Batch Video Testing**
### **Data-Driven Validation**

- **Dataset**: Validated against **7 Subjects** from UBFC-rPPG.
- **Findings**:
    - **Alert Subjects**: Correctly identified (Avg Prob: **0.40**).
    - **Drowsy Subjects**: Correctly triggered **Fatigued/CRITICAL** states (Avg Prob: **0.56 - 0.81**).
- **Automation**: Created `test_multiple_videos.py` for automated batch verification.

---

# **System Performance & Accuracy**

| Metric | Result | Status |
| :--- | :--- | :--- |
| **Heart Rate Accuracy** | **6.03 BPM** (MAE) | Benchmark Verified |
| **Fusion Model Loss** | **0.0035** (MSE) | Trained & Verified |
| **Processing Speed** | **30 FPS** (Locked) | Asynchronous Worker |
| **Tracking Reliability** | **99%+** Face Detection | MediaPipe Robust |

---

# **Conclusion & Next Steps**

- [ ] **Blink Detection**: Peak analysis on normalized EAR streams.
- [ ] **Model Quantization**: Convert `.h5` to **TensorFlow Lite (.tflite)**.
- [ ] **Audio Feedback**: Localized voice/beep alerts for Critical states.

---

# **Questions?**
### AI Driving Monitoring System
**Calibrated. Responsive. Robust.**
