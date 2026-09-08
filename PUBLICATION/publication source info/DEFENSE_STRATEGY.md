# [DEFENSE GUIDE] AI Driving Monitoring System: Top 10 Questions & Answers

This guide is designed to help you prepare for your oral defense at Tatung University. These questions target the "Technical Core" and "Novelty" of your thesis.

---

### 1. "Why did you choose a 'Hard-Switching' mechanism for ROI selection instead of just weighting (Soft Attention) all facial patches?"
*   **The Answer**: Weighting (Soft Attention) like in Bobbia et al. (2017) averages noise from shadowed regions into the final signal. My "Hard-Switching" pivots to the single best ROI, completely discarding noise. This ensures the BiLSTM receives the cleanest possible physiological stream, which is critical for calculating sensitive features like HRV Delta.

### 2. "Explain the biological basis for using HRV Delta as a predictive indicator. Why is it better than just Heart Rate?"
*   **The Answer**: Heart Rate (HR) is a reactive measure of physical exertion. Heart Rate Variability (HRV), specifically SDNN, reflects the Sympathovagal balance. As a driver becomes drowsy, parasympathetic dominance increases, causing a specific "slope" or decline in HRV. Capturing the *rate of change* (HRV Delta) allows the system to predict sleep onset minutes before the driver physically blinks (PERCLOS).

### 3. "Your Multi-Head Attention (MHA) has 4 heads. What is the benefit of having multiple heads for this specific task?"
*   **The Answer**: Each head can attend to a different subspace of the 7-feature vector. For example, Head 1 may focus on behavioral "spikes" like yawning (MAR), while Head 2 monitors the long-term physiological trend of HRV Delta. This parallel processing allows the model to catch multiple types of fatigue signs simultaneously.

### 4. "Your Validation MAE is 0.0332. This is extremely low. Is your model overfitting to the NTHU-DDD dataset?"
*   **The Answer**: To prevent overfitting, I utilized a 50-subject "Universal Benchmark" (UBFC-rPPG) which the model never saw during training. Additionally, I implemented Dropout layers (0.5) and Batch Normalization. The low MAE is a result of the 77% improvement achieved by fusing internal physiological trends with external behavioral signs.

### 5. "Why did Subjects 11 and 20 exhibit such high error (MAE > 39 BPM)?"
*   **The Answer**: These were "Boundary Cases." Subject 20 had extreme non-rigid motion (talking/laughing), which violates the POS skin-stability assumption. Subject 11 had pixel saturation (clipping at 255) from a low-angle light source. I documented these to show that the system is fail-operational: the Attention mechanism detected the failure and pivoted to alternative features.

### 6. "How did you achieve 30 FPS on a CPU without a GPU?"
*   **The Answer**: Optimization was achieved through: 1) Asynchronous Multi-threading using the `AIBackgroundWorker` to decouple inference from capture. 2) Spatial downsampling of ROIs to 64x64 pixels. 3) Pruning redundant MediaPipe landmarks to reduce the memory footprint.

### 7. "Explain the Modified Beer-Lambert Law and how it relates to your rPPG extraction."
*   **The Answer**: The law explains that light absorption is proportional to the concentration of hemoglobin. Since blood volume in the dermis varies rhythmically with the heart, the light absorption also varies. We focus on the Green channel (530nm) because it has the highest absorption peak for oxyhemoglobin and optimal tissue penetration depth.

### 8. "How does your system handle drivers wearing medical masks or sunglasses?"
*   **The Answer**: This is where Multi-modal fusion is strongest. If a mask covers the cheeks, the ROI-Switching Attention pivots to the Forehead. If sunglasses block the eyes (PERCLOS), the MHA mechanism automatically increases the weight of the rPPG signal and HRV Delta to maintain monitoring integrity.

### 9. "What is the significance of the attention heatmaps (XAI) for an automotive company?"
*   **The Answer**: For safety-critical systems, "System Trust" is essential. The heatmaps provide a transparent audit trail. Instead of a "Black Box" alert, the system can prove that it triggered a warning because of a specific 2-second drop in HRV or an increase in EAR closure, which is vital for accident investigation.

### 10. If you had 6 more months, what is the #1 thing you would improve?
*   **The Answer**: I would integrate Near-Infrared (NIR) sensor support. This would remove the dependency on visible light, enabling the system to perform with the same 0.0332 MAE during night-driving or in completely dark cabin environments.

---

### 11. "How can you justify comparing your MAE results with other researchers if they used different datasets?"
*   **The Answer**: "I acknowledge that absolute MAE numbers are dataset-dependent. However, the true significance of my research is the **Relative Improvement**. My system achieved a **77% reduction in error** compared to its own baseline, which is significantly higher than the improvements reported in prior literature. Furthermore, to prove that my results weren't just 'lucky' on one dataset, I performed **Universal Validation** on the 50-subject UBFC-rPPG dataset, which confirms that the MHA-BiLSTM architecture generalizes effectively across different environments and subjects."

