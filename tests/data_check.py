import os, sys
sys.path.append(os.getcwd())
import json
import numpy as np
import pandas as pd

def main():
    # Check All Samples, Consistency
    
    root = "data/open-nuro-dataset/dataset"
    
    freq_li = []
    task_li = []
    duration_li = []
    channels_li = []
    
    for i in range(1, 88+1):
        path = f"{root}/sub-{i:03d}/eeg/sub-{i:03d}_task-eyesclosed"
        with open(f"{path}_eeg.json", 'r') as f:
            metadata = json.load(f)
            freq_li.append(metadata['SamplingFrequency'])
            task_li.append(metadata['TaskName'])
            duration_li.append(metadata['RecordingDuration'])
            
        channels = pd.read_csv(f"{path}_channels.tsv", sep='\t')
        channels = channels['name'].tolist()
        channels_li.append(channels)
            
    freq_li = np.unique(freq_li)
    task_li = np.unique(task_li)
    channels_li = np.unique(channels_li)
    
    dur_mean, dur_std = np.mean(duration_li), np.std(duration_li)
    
    print(freq_li) # [500], All Samples 500Hz
    print(task_li) # ['eyeclosed'], Same Task
    print(channels_li) # Same Channel Settings, 10-20 System
    print(f"Duration Mean: {dur_mean:.2f}s, Duration Std: {dur_std:.2f}s") # Duration Mean, Standard Deviation
    
if __name__ == '__main__':
    main()