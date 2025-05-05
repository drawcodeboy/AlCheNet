import torch
from torch import nn
from ..ChannelwiseEncoder.convnet import ConvNet
from ..mlp import MLP

class BaseNet(nn.Module):
    # ConvNet + MLP
    def __init__(self,
                 enc_channels:int,
                 enc_window_size:int,
                 enc_seq_len:int,
                 enc_node_dim:int,
                 mlp_in_features:int,
                 mlp_hidden_features:int,
                 mlp_class_num:int):
        super().__init__()

        self.encoder = ConvNet(channels=enc_channels,
                               window_size=enc_window_size,
                               seq_len=enc_seq_len,
                               node_dim=enc_node_dim,
                               mlp_mode='node_rep')
        
        self.mlp = MLP(in_features=mlp_in_features,
                       hidden_features=mlp_hidden_features,
                       class_num=mlp_class_num)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.mlp(x)
        
        return x
    
    @classmethod
    def from_config(cls, cfg):
        return cls(enc_channels=cfg['enc_channels'],
                   enc_window_size=cfg['enc_window_size'],
                   enc_seq_len=cfg['enc_seq_len'],
                   enc_node_dim=cfg['enc_node_dim'],
                   mlp_in_features=cfg['mlp_in_features'],
                   mlp_hidden_features=cfg['mlp_hidden_features'],
                   mlp_class_num=cfg['mlp_class_num'])