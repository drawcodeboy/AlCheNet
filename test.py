import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

root = "data/open-nuro-dataset/dataset/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"

raw = mne.io.read_raw_eeglab(root, preload=True, verbose=False)

print(raw.info['ch_names'])

raw.filter(l_freq=1.0, h_freq=40.0)

metadata = pd.read_csv("data/open-nuro-dataset/dataset/participants.tsv", sep='\t')

raw.plot()
data = raw._data
print(data.shape)

plt.savefig('./test.png')