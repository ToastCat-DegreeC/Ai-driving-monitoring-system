import numpy as np
from utils.signal_processing import calculate_sdnn

def test_hrv():
    print("Testing HRV calculation...")
    # Create a dummy periodic signal (30 Hz, 1 beat per second = 60 BPM)
    fs = 30
    duration = 10 # seconds
    t = np.linspace(0, duration, fs * duration)
    # Simple sine wave with peaks every 30 samples
    signal = np.sin(2 * np.pi * 1.0 * t)
    
    sdnn = calculate_sdnn(signal, fs=fs)
    print(f"Calculated SDNN for perfect 60 BPM signal: {sdnn:.2f} ms")
    
    # Create a jittery signal
    # Peaks at 30, 65, 92, 128...
    jitter_signal = np.zeros(300)
    peaks = [30, 65, 92, 128, 160, 195, 230, 265]
    for p in peaks:
        jitter_signal[p] = 1.0
    # Smooth it slightly
    jitter_signal = np.convolve(jitter_signal, [0.2, 0.5, 1.0, 0.5, 0.2], mode='same')
    
    sdnn_jitter = calculate_sdnn(jitter_signal, fs=fs)
    print(f"Calculated SDNN for jittery signal: {sdnn_jitter:.2f} ms")
    
    if sdnn_jitter > 0:
        print("SUCCESS: HRV detection logic is functional.")
    else:
        print("FAILURE: HRV detection logic returned 0.")

if __name__ == "__main__":
    test_hrv()
