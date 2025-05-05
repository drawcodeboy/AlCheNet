import torch
from torch import nn
from einops import rearrange

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels: int,
                 kernel_row: int, # 1 or 5, individual or integrate
                 norm_dim: int,
                 window_size:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(kernel_row, window_size),
                               groups=in_channels,
                               padding=(0, window_size//2))
        if norm_dim == 2:
            self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        elif norm_dim == 1:
            self.norm1 = nn.BatchNorm1d(num_features=out_channels)
            
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(1, window_size),
                               groups=in_channels,
                               padding=(0, window_size//2))
        if norm_dim == 2:
            self.norm2 = nn.BatchNorm2d(num_features=out_channels)
        elif norm_dim == 1:
            self.norm2 = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.kernel_row = kernel_row
        
    def forward(self, x):
        x = self.conv1(x)
        if self.kernel_row == 5:
            x = rearrange(x, 'b c 1 l -> b c l')
            
        x = self.relu(self.norm1(x))
        x = self.relu(self.norm2(self.conv2(x)))
        return x