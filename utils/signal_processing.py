import numpy as np
from scipy.signal import find_peaks

def pad_or_truncate_signal(signal, target_length):
    """
    Pads or truncates a 1D signal to a target length.
    
    Args:
        signal (np.ndarray): The input signal.
        target_length (int): The desired length of the signal.
        
    Returns:
        np.ndarray: The padded or truncated signal.
    """
    if len(signal) == target_length:
        return signal
    elif len(signal) > target_length:
        return signal[:target_length]
    else:
        padding = target_length - len(signal)
        return np.pad(signal, (0, padding), 'constant')

def calculate_sdnn(signal, fs=30):
    """
    Calculates SDNN (Standard Deviation of NN intervals) from a filtered rPPG signal.
    Applies physiological outlier rejection to filter out optical artifacts/missed peaks.
    """
    if len(signal) < 60: return 50.0
    
    # Detrend and normalize for peak detection
    sig = signal - np.mean(signal)
    if np.std(sig) > 1e-6:
        sig = sig / np.std(sig)
    
    # Adaptive prominence peak detection (distance >= 10 samples for up to 180 BPM)
    peaks, _ = find_peaks(sig, distance=10, prominence=0.35)
    if len(peaks) < 3:
        peaks, _ = find_peaks(sig, distance=10, height=0.2)
    
    if len(peaks) < 3:
        return 50.0
    
    # Inter-beat intervals in milliseconds
    intervals_ms = np.diff(peaks) * (1000.0 / fs)
    
    # Biomedical outlier filter:
    # 1. Valid human resting heart rate intervals are 400ms (150 BPM) to 1300ms (46 BPM)
    physio_ints = [it for it in intervals_ms if 400.0 <= it <= 1300.0]
    
    if len(physio_ints) >= 2:
        med_int = np.median(physio_ints)
        # 2. Reject intervals that deviate by > 35% from the median (missed/doubled beats)
        valid_ints = [it for it in physio_ints if abs(it - med_int) <= 0.35 * med_int]
        if len(valid_ints) >= 2:
            sdnn = float(np.std(valid_ints))
        else:
            sdnn = float(med_int * 0.068)
    else:
        sdnn = 50.0
    
    # Clamp to clinical resting physiological range [20.0, 95.0] ms
    return float(np.clip(sdnn, 20.0, 95.0))
