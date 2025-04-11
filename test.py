import mne
import matplotlib as plt

root = "data/physionet.org/files/sleep-edf/1.0.0/sc4002e0.edf"

raw = mne.io.read_raw_edf(root, preload=True, verbose=False)
print(raw.info['ch_names'])