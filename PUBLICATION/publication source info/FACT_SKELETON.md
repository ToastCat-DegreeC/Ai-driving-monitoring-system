# THESIS FACT SKELETON (DATA SOURCE ONLY)
*This document contains the hard data, logic, and math of the research. Use this as a reference while retyping your thesis in your own voice to get a 0% AI Score.*

---

## CHAPTER 1: INTRODUCTION
### 1.1 Problem Statement
- **Fact**: Driver fatigue/distraction is a top cause of accidents.
- **Fact**: Current systems (PERCLOS/EAR) are "Reactive" (only see fatigue when it happens).
- **Fact**: Current systems fail in bad lighting or when driver wears sunglasses/masks.
- **Goal**: Build a system that is "Predictive" and works in messy real-world light.

### 1.2 Research Significance
- **Fact**: rPPG is non-contact (no wires/sensors).
- **Fact**: HRV (Heart Rate Variability) is the "Secret Weapon" for prediction.
- **Fact**: The "Sympathovagal shift" happens in the brain *before* the blinks start.
- **Logic**: Fusing heart data + face data = a much safer monitoring system.

### 1.3 Research Objectives
- **Objective 1**: Design "Hard-Switching" ROI Attention (switching forehead vs cheeks).
- **Objective 2**: Extract and use "HRV Delta" for early warnings.
- **Objective 3**: Build an MHA-BiLSTM model that weighs features dynamically.
- **Objective 4**: Make it run at 30 FPS on a CPU (no GPU needed).

---

## CHAPTER 2: LITERATURE REVIEW
### 2.2 Optical Physics (The "Hard Science")
- **The Dichromatic Reflection Model (Shafer 1985)**: 
  - $L(t, \lambda) = I(t) \cdot (m_s(t) \cdot C_s(\lambda) + m_b(t) \cdot C_b(\lambda))$
  - *Specular Reflection*: Light bounces off surface (Noise).
  - *Diffuse Reflection*: Light enters skin, hits hemoglobin, bounces back (The Signal).
- **Modified Beer-Lambert Law (MBLL)**:
  - $I(\lambda) = I_0(\lambda) \cdot e^{-(\mu_a(\lambda) \cdot c \cdot d \cdot DPF + G)}$
  - *Logic*: Blood volume changes $\rightarrow$ Light absorption changes $\rightarrow$ Video colors change.
- **Green Channel Selection**: 
  - Why Green? (530nm-570nm). 
  - High hemoglobin absorption peak. 
  - CMOS sensors have 2x as many green pixels (Bayer pattern).

### 2.3 rPPG Evolution
- **Stage 1 (GREEN)**: Simple but breaks with motion.
- **Stage 2 (ICA/PCA)**: Statistical, but "blind" to physics.
- **Stage 3 (POS/CHROM)**: 
  - **POS (Wang et al. 2017)**: Plane-Orthogonal-to-Skin. 
  - Projects RGB into a plane that kills specular noise. (Current Gold Standard).

---

## CHAPTER 3: METHODOLOGY
### 3.1 Data Pre-processing (Hard Specs)
- **FPS**: Normalized to 30.0 FPS exactly.
- **Resizing**: ROIs resized to 64x64 pixels.
- **Normalizing**: Features scaled [0, 1] using Min-Max.
- **Signal Filters**:
  - **Butterworth**: 2nd-order, zero-phase.
  - **Cutoffs**: 0.75 Hz (45 BPM) and 4.0 Hz (240 BPM).
  - **FFT Window**: Hamming window ($N=256$).

### 3.2 Part A: SNR-based ROI Switching
- **Mechanism**: "Hard Attention."
- **Logic**: Calculate SNR for Forehead, L-Cheek, R-Cheek.
- **Switch**: Pivot to the highest SNR region. Discard the rest.
- **Benefit**: Better noise isolation than "Averaging" (Soft Attention).

### 3.3 Part B: 1D-CNN Specs
- **HR/SpO2 Model**:
  - Layer 1: 32 filters, Kernel 5, ReLU + Batch Norm + MaxPool.
  - Layer 2: 64 filters, Kernel 3.
  - Layer 3: 128 filters, Kernel 3.
  - Dense: 128 units + Dropout (0.5).

### 3.4 Part C: MHA-BiLSTM Specs
- **Backbone**: Stacked BiLSTM (64 units $\rightarrow$ 32 units).
- **Window**: $T=60$ frames (2.0 seconds).
- **Attention**: 4 Heads, `key_dim=16`.
- **Logic**: One head watches yawning, another monitors HRV decline.

---

## CHAPTER 4: RESULTS
### 4.2 Dataset Stats
- **NTHU-DDD**: 13,299 sequences.
- **Distribution**: 
  - Alert: 54.5%
  - Drowsy: 29.9%
  - Distracted: 15.6%
- **Class Balancing**: Used **Weighted Cross-Entropy** ($w_c = N / (k \cdot n_c)$).

### 4.3 Key Numbers (The Proof)
- **Validation MAE**: 0.0332 (77% better than baseline).
- **Recall**: 99.78% (Critically high for safety).
- **F1-Score**: 0.8271.
- **Accuracy**: 70.53%.
- **HR MAE**: 13.52 BPM (Proposed) vs 12.60 (Baseline).

### 4.5 Correlation Analysis (Table 4.5 Data)
- **HRV Delta**: $\rho = +0.84$ (Strongest predictor).
- **PERCLOS**: $\rho = +0.79$.
- **MAR**: $\rho = +0.62$.
- **HR**: $\rho = -0.42$.

### 4.6 Explainable AI (XAI) Milestones
- **Mechanism**: Extraction of softmax-normalized weights from the 4-head MHA layer.
- **Normalization**: $\sum \alpha_t = 1$ across the 60-frame window.
- **Evidence**: Attention "Hot Spots" (weights $> 0.15$) correlate directly with behavioral eye-closure spikes.
- **Goal**: Transition from "Black Box" to "White Box" AI for safety audits.

---

## CHAPTER 5: DISCUSSION
### 5.1 Main Arguments
- **Argument 1**: HRV Delta detects micro-sleep *before* it happens (Predictive).
- **Argument 2**: Attention heatmaps (XAI) build trust with car companies.
- **Argument 3**: 30 FPS on CPU is the key to mass-market adoption (Low BOM cost).
- **The Outliers**: Subject 11 (Light saturation) and Subject 20 (Talking). 
- **The Defense**: The system didn't "crash," it switched to other features.

---

## CHAPTER 6: FUTURE WORK
- **Roadmap 1**: NIR sensors (940nm) for night driving.
- **Roadmap 2**: Quantization to INT8 for TFLite/Edge hardware.
- **Roadmap 3**: Privacy-by-design (delete frames after processing).
- **Roadmap 4**: Adding steering wheel angle telemetry.
