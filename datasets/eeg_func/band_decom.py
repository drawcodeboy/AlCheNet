from scipy.signal import sosfiltfilt, butter

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

def decompose_eeg_waves(eeg_signal, fs):
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 40),
    }
    filtered = {}
    for band, (low, high) in bands.items():
        filtered[band] = bandpass_filter(eeg_signal, low, high, fs)
    return filtered  # dict: {'delta': np.array, ...}