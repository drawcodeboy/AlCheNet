import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import mne
from einops import rearrange
from scipy.signal import welch
from torch_geometric.utils import dense_to_sparse
from .eeg_func.band_decom import decompose_eeg_waves
from itertools import combinations

import os, sys
import time
import random

class NuroDataset(Dataset):
    '''
        Need to convert Numpy Format by using data2np.py
        Command: python datasets/data2np.py
    '''
    def __init__(self,
                 root:str="data/open-nuro-dataset/dataset",
                 mode:str='train',
                 pathology_graph:bool=True,
                 parcellation:str='LR',
                 edge_w:float=0.5):
        super().__init__()
        
        self.root = root
        if mode not in ['train', 'val', 'test']:
            raise Exception(f"Dataset mode {mode} is not supported.")
        self.mode = mode
        self.metadata = pd.read_csv(f"{root}/participants.tsv", sep='\t')
        
        self.data_li = []
        self.label_map = {
            'A': 0, # Alzheimer's disease (AD group)
            'C': 1, # Frontotemporal Dementia (FTD group)
            'F': 2, # healthy subjects (CN group)
        }
        self.test_li = [33, 34, 35, 36, 62, 63, 64, 65, 85, 86, 87, 88] # Each class, 4 Participants
        self.test_li = list(map(lambda x: f"sub-{x:03d}", self.test_li))
        
        self.pathology_graph = pathology_graph
        self.parcellation = parcellation
        self.w_adj = torch.ones((19, 19)) * 0.5
        self._build_w_adj()
        self._check()
        
        random.seed(42)
        random.shuffle(self.data_li)
        
        # Train, Validation Split 90:10
        train_size = int(len(self.data_li) * 0.9)
        if self.mode == 'train':
            self.data_li = self.data_li[:train_size]
        elif self.mode == 'val':
            self.data_li = self.data_li[train_size:]
        
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        freq_path, sample_path, adj_path, label = self.data_li[idx]
        freq = np.load(freq_path).astype(np.float32)
        sample = np.load(sample_path).astype(np.float32)
        adj = np.load(adj_path).astype(np.float32)
        adj = torch.tensor(adj, dtype=torch.float32)
        
        if self.pathology_graph == True:
            adj = adj * self.w_adj
        
        edge_index, edge_weight = dense_to_sparse(adj)
        
        sample = torch.tensor(sample, dtype=torch.float32)
        freq = torch.tensor(freq, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)
        
        x = {'freq': freq,
             'sample': sample,
             'edge_index': edge_index,
             'edge_weight': edge_weight}
        return x, label
    
    def _build_w_adj(self):
        channels_str = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", 
                        "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
        
        channels_li = []
        if self.parcellation == 'ALL':
            channels_li.append(list(combinations([0, 1, 2, 3, 10, 11, 16], 2)))
            channels_li.append(list(combinations([4, 5, 17], 2)))
            channels_li.append(list(combinations([6, 7, 18], 2)))
            channels_li.append(list(combinations([12, 13, 14, 15], 2)))
            channels_li.append(list(combinations([8, 9], 2)))
        
        for channels in channels_li:
            for comb in channels:
                self.w_adj[comb[0], comb[1]] = 1.
                # print(channels_str[comb[0]], channels_str[comb[1]])
            # print("=====")
        # print(self.w_adj)
        
    def _check(self):
        participants_li = self.metadata['participant_id'].tolist()
        np_root = f"{self.root}/np-format"
        
        for participant_id in sorted(os.listdir(np_root)):
            if self.mode == 'train' or self.mode == 'val':
                if participant_id in self.test_li: continue
            else:
                if participant_id not in self.test_li: continue
            
            participant_path = f"{np_root}/{participant_id}"
            for freq_path in sorted(os.listdir(participant_path)):
                if freq_path[:4] != "freq":
                    continue
                adj_path = freq_path.replace('freq', 'adj')
                sample_path = freq_path.replace('freq', 'sample')
                freq_path = f"{participant_path}/{freq_path}"
                adj_path = f"{participant_path}/{adj_path}"
                sample_path = f"{participant_path}/{sample_path}"
                label = self.metadata[self.metadata['participant_id'] == participant_id]['Group'].values.item()
                label = self.label_map[label]
                
                self.data_li.append([freq_path, sample_path, adj_path, label])
                
    @classmethod
    def from_config(cls, cfg):
        return cls(root=cfg['root'],
                   mode=cfg['mode'],
                   pathology_graph=cfg.get('pathology_graph'),
                   parcellation=cfg.get('parcellation'),
                   edge_w=cfg.get('edge_w'))

if __name__ == '__main__':
    train_ds = NuroDataset()
    val_ds = NuroDataset(mode='val')
    test_ds = NuroDataset(mode='test')