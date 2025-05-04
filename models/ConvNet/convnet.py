import torch
from torch import nn
from einops import rearrange

# from ..mlp import MLP
import os, sys
sys.path.append(os.getcwd())
from models.mlp import MLP

class ConvNet(nn.Module):
    def __init__(self,
                 channels:int=19,
                 window_size:int=5,
                 seq_len:int=50,
                 node_dim:int=128,
                 mlp_mode:str='conv_cls', # conv_cls or node_rep
                 mlp_hidden_features:int=64,
                 class_num:int=3): 
        super().__init__()
        
        # Depthwise Convolution
        # Feature Extraction each waves
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=8*channels,
                               kernel_size=(1, window_size),
                               groups=channels,
                               padding=(0, window_size//2))
        self.norm1 = nn.BatchNorm2d(num_features=8*channels)
        
        # Integrate Waves
        self.conv2 = nn.Conv2d(in_channels=8*channels,
                               out_channels=16*channels,
                               kernel_size=(5, window_size),
                               groups=channels,
                               padding=(0, window_size//2))
        self.norm2 = nn.BatchNorm1d(num_features=16*channels)
        
        self.li = nn.Linear(in_features=seq_len,
                            out_features=node_dim)
        
        self.relu = nn.ReLU()
        
        
        self.mlp = None
        self.mlp_mode = mlp_mode
        if self.mlp_mode == 'conv_cls':
            self.mlp = MLP(in_features=node_dim,
                           hidden_features=mlp_hidden_features,
                           class_num=class_num)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = rearrange(x, 'b c 1 l -> b c l')
        x = self.norm2(x)
        x = self.relu(x)
        
        B, C, L = x.size()
        x = x.view(B, 16, 19, L)
        x = x.mean(dim=1)
        
        x = self.li(x)
        x = self.relu(x)
        
        if self.mlp_mode == 'conv_cls':
            x = x.mean(dim=1) # We should consider here, inter-channels interaction is just mean.
            x = self.mlp(x)
            return x
        elif self.mlp_mode == 'node_rep':
            return x
    
    @classmethod
    def from_config(cls, cfg):
        return cls(channels=cfg['channels'],
                   window_size=cfg['window_size'],
                   seq_len=cfg['seq_len'],
                   node_dim=cfg['node_dim'],
                   mlp_mode=cfg['mlp_mode'], # conv_cls or node_rep
                   mlp_hidden_features=cfg['mlp_hidden_features'],
                   class_num=cfg['class_num'])
    
if __name__ == '__main__':
    model = ConvNet()
    
    x = torch.randn(32, 19, 5, 50) # (Batch Size, Channels, Waves, Length)
    x = model(x)
    print(x.shape)