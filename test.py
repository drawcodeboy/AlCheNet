import mne
import matplotlib as plt

root = "data/open-nuro-dataset/dataset/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"

raw = mne.io.read_raw_eeglab(root, preload=True, verbose=False)
print(raw.info['ch_names'])

print(raw)