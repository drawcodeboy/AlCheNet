import sys, os
sys.path.append(os.getcwd())
import numpy as np
import mne
import pandas as pd
from scipy.signal import resample, welch
from eeg_func.band_decom import decompose_eeg_waves
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from multiprocessing import Pool

def compute_dtw(pair):
    i, j, sample = pair
    dist, _ = fastdtw(sample[i], sample[j])
    return (i, j, dist)

# 병렬 처리 함수
def parallel_dtw(sample):
    adj = np.zeros((sample.shape[0], sample.shape[0]))

    # 모든 (i, j) 쌍을 계산하기 위한 인덱스 목록 생성
    indices = [(i, j, sample) for i in range(sample.shape[0]) for j in range(i + 1, sample.shape[0])]

    # Pool을 사용하여 병렬 처리
    with Pool() as pool:
        results = pool.map(compute_dtw, indices)

    # 결과로부터 distance 값을 adjacency matrix에 넣기
    for i, j, dist in results:
        adj[i, j] = dist
        adj[j, i] = dist

    return adj

def main():
    root = "data/open-nuro-dataset/dataset"
    save_path = f"{root}/np-format"
    os.makedirs(save_path, exist_ok=True)
    
    seconds = 10 # Choose data length
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
        data = mne.filter.filter_data(data, sfreq=fs, l_freq=1, h_freq=40, verbose=False) # 1~40Hz Band Pass Filtering
        
        for index, start_point in enumerate(range(0, data.shape[1], length), start=1): # length//2 for overlapping samples
            if (start_point+length) > data.shape[1]: # drop last sample
                break
            sample = data[:, start_point:start_point+length]
            if factor == 1:
                pass
            else:
                sample = resample(sample, num=length//factor, axis=1) # Down Sampling
            
            # Node Vector for ConvNet, Graph Network (Channels, Frequecies bins, Densitiy)
            channels_li = []
            for i in range(0, sample.shape[0]): # for Channels
                channel_sample = sample[i]
                decomposed_channel_sample = decompose_eeg_waves(channel_sample, fs=fs//factor)
                Pxx_norm_li = []
                for key in decomposed_channel_sample.keys(): # for Waves
                    f, Pxx = welch(decomposed_channel_sample[key], fs=fs//factor, nperseg=fs//factor) # Resolution 1Hz -> segment=1
                    Pxx = Pxx[:50+1]
                    
                    # Log Transform + Min-Max Scaling
                    Pxx_log = np.log1p(Pxx)  # log(1 + x), 0 대응 가능
                    Pxx_norm = (Pxx_log - Pxx_log.min()) / (Pxx_log.max() - Pxx_log.min())
                    Pxx_norm_li.append(Pxx_norm)
                Pxx_norm = np.vstack(tuple(Pxx_norm_li))
                channels_li.append(Pxx_norm)
            freq = np.stack(tuple(channels_li), axis=0)
            np.save(f"{save_sample_path}/freq_{index:04d}.npy", freq)
            
            np.save(f"{save_sample_path}/sample_{index:04d}.npy", sample)
                
            # Adjacency Matrix for Graph Network (DTW)
            adj = parallel_dtw(sample) # Multi-Processing

            sigma = np.std(adj)
            adj = np.exp(-(adj ** 2)/(2 * sigma ** 2))
            np.save(f"{save_sample_path}/adj_{index:04d}.npy", adj)
            
        print(f"\rData Processing... ({idx/len(participants_li)*100:.2f}%)", end='')
    print()

if __name__ == '__main__':
    main()