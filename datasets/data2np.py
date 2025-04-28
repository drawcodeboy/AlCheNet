import sys, os
sys.path.append(os.getcwd())
import numpy as np
import mne
import pandas as pd

def main():
    # It takes about 30 seconds
    root = "data/open-nuro-dataset/dataset"
    save_path = f"{root}/np-format"
    os.makedirs(save_path, exist_ok=True)
    
    seconds = 10 # Choose data length
    fs = 500
    length = fs * seconds
    
    metadata = pd.read_csv(f"{root}/participants.tsv", sep='\t')
    participants_li = metadata['participant_id'].tolist()
    
    for participant_id in participants_li:
        path = f"{root}/{participant_id}/eeg/{participant_id}_task-eyesclosed_eeg.set"
        save_sample_path = f"{save_path}/{participant_id}"
        os.makedirs(save_sample_path, exist_ok=True)
            
        raw = mne.io.read_raw_eeglab(path, preload=True)
        data = raw._data
        
        for index, start_point in enumerate(range(0, data.shape[1], length), start=1):
            if (start_point+length) > data.shape[1]: # drop last sample
                break
            sample = data[:, start_point:start_point+length]
            np.save(f"{save_sample_path}/sample_{index:04d}.npy", sample)

if __name__ == '__main__':
    main()