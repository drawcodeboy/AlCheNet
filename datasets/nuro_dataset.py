import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import mne
from einops import rearrange

import os
import time
import random

class NuroDataset(Dataset):
    '''
        Need to convert Numpy Format by using data2np.py
        Command: python datasets/data2np.py
    '''
    def __init__(self,
                 root:str="data/open-nuro-dataset/dataset",
                 mode:str='train'):
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
        sample_path, label = self.data_li[idx]
        sample = np.load(sample_path).astype(np.float32)
        
        # NEED PRE-PROCESSING
        sample = torch.tensor(sample)
        sample = rearrange(sample, 'C L -> L C') # C = Channels, L = Length
        
        return sample, label
        
    def _check(self):
        participants_li = self.metadata['participant_id'].tolist()
        np_root = f"{self.root}/np-format"
        
        for participant_id in sorted(os.listdir(np_root)):
            if self.mode == 'train' or self.mode == 'val':
                if participant_id in self.test_li: continue
            else:
                if participant_id not in self.test_li: continue
            
            participant_path = f"{np_root}/{participant_id}"
            for sample_path in sorted(os.listdir(participant_path)):
                sample_path = f"{participant_path}/{sample_path}"
                label = self.metadata[self.metadata['participant_id'] == participant_id]['Group'].values.item()
                label = self.label_map[label]
                
                self.data_li.append([sample_path, label])
                
    @classmethod
    def from_config(cls, cfg):
        return cls(root=cfg['root'],
                   mode=cfg['mode'])

if __name__ == '__main__':
    train_ds = NuroDataset()
    val_ds = NuroDataset(mode='val')
    test_ds = NuroDataset(mode='test')