import numpy as np
from scipy import signal
from scipy.stats import entropy

def bandpass_filter(data, lowcut=0.5, highcut=45, fs=128, order=5):
    """Apply Butterworth band-pass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)
    return filtered

def notch_filter(data, freq=60, fs=128, Q=30):
    """Remove 60 Hz power line interference"""
    b, a = signal.iirnotch(freq, Q, fs)
    filtered = signal.filtfilt(b, a, data)
    return filtered

def compute_hjorth_parameters(sig):
    """Compute Hjorth Activity, Mobility, Complexity"""
    diff1 = np.diff(sig)
    diff2 = np.diff(diff1)
    
    activity = np.var(sig)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
    
    return activity, mobility, complexity

def extract_features(data, fs=128):
    """Extract all features from EEG window"""
    # Already have band powers in dataset, but keeping function for completeness
    features = {
        'mean': np.mean(data),
        'std': np.std(data)
    }
    return features

if __name__ == "__main__":
    print("Signal processing module ready")
    
    # Test
    test_signal = np.random.randn(256)
    filtered = bandpass_filter(test_signal)
    print(f"Test successful: {len(filtered)} samples")
