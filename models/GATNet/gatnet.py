import torch
from torch import nn
from ..ChannelwiseEncoder.convnet import ConvNet
from .graphblock import GraphBlock

class GATNet(nn.Module):
    def __init__(self,
                 enc_channels:int,
                 enc_window_size:int,
                 enc_seq_len:int,
                 enc_node_dim:int,
                 in_channels:int,
                 hidden_channels:int,
                 out_channels:int,
                 n_layer:int,
                 mlp_hidden_features:int,
                 mlp_class_num:int):
        super().__init__()

        self.encoder = ConvNet(channels=enc_channels,
                               window_size=enc_window_size,
                               seq_len=enc_seq_len,
                               node_dim=enc_node_dim,
                               mlp_mode='node_rep')
        
        self.graphblock = GraphBlock(in_channels=in_channels,
                                     hidden_channels=hidden_channels,
                                     out_channels=out_channels,
                                     n_layer=n_layer,
                                     mlp_hidden_features=mlp_hidden_features,
                                     mlp_class_num=mlp_class_num)
        
    def forward(self, x):
        freq = x['freq']
        freq = self.encoder(freq)
        x['freq'] = freq
        x = self.graphblock(x)
        return x
    
    @classmethod
    def from_config(cls, cfg):
        return cls(enc_channels=cfg['enc_channels'],
                   enc_window_size=cfg['enc_window_size'],
                   enc_seq_len=cfg['enc_seq_len'],
                   enc_node_dim=cfg['enc_node_dim'],
                   in_channels=cfg['in_channels'],
                   hidden_channels=cfg['hidden_channels'],
                   out_channels=cfg['out_channels'],
                   n_layer=cfg['n_layer'],
                   mlp_hidden_features=cfg['mlp_hidden_features'],
                   mlp_class_num=cfg['mlp_class_num'])