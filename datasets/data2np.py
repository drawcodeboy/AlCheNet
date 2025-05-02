import sys, os
sys.path.append(os.getcwd())
import numpy as np
import mne
import pandas as pd
from scipy.signal import resample

def main():
    # It takes about 30 seconds
    root = "data/open-nuro-dataset/dataset"
    save_path = f"{root}/np-format"
    os.makedirs(save_path, exist_ok=True)
    
    seconds = 30 # Choose data length
    fs = 500
    length = fs * seconds
    factor = 5 # Down Sampling to (fs / factor)Hz
    
    metadata = pd.read_csv(f"{root}/participants.tsv", sep='\t')
    participants_li = metadata['participant_id'].tolist()
    
    for idx, participant_id in enumerate(participants_li, start=1):
        path = f"{root}/{participant_id}/eeg/{participant_id}_task-eyesclosed_eeg.set"
        save_sample_path = f"{save_path}/{participant_id}"
        os.makedirs(save_sample_path, exist_ok=True)
            
        raw = mne.io.read_raw_eeglab(path, preload=True)
        data = raw._data
        data = mne.filter.filter_data(data, sfreq=fs, l_freq=1, h_freq=40, verbose=False)
        
        '''
        import matplotlib.pyplot as plt
        plt.subplot(3, 1, 1)
        y = data[0, :1000]
        x = np.arange(0, len(y))
        plt.plot(x, y)
        
        plt.subplot(3, 1, 2)
        data = mne.filter.filter_data(data, sfreq=fs, l_freq=1, h_freq=40, verbose=False)
        y = data[0, :1000]
        x = np.arange(0, len(y))
        plt.plot(x, y)
        
        plt.subplot(3, 1, 3)
        y = data[:, :1000]
        y = resample(y, num=100, axis=1)
        y = y[0]
        x = np.arange(0, len(y))
        plt.plot(x, y)
        
        plt.tight_layout()
        plt.savefig('./EEG.png')
        sys.exit()
        '''
        
        for index, start_point in enumerate(range(0, data.shape[1], length), start=1):
            if (start_point+length) > data.shape[1]: # drop last sample
                break
            sample = data[:, start_point:start_point+length]
            sample = resample(sample, num=length//factor, axis=1)
            np.save(f"{save_sample_path}/sample_{index:04d}.npy", sample)
        print(f"\rData Processing... ({idx/len(participants_li)*100:.2f}%)", end='')
    print()

if __name__ == '__main__':
    main()