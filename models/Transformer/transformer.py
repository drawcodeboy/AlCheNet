import torch
from torch import nn
from ..ChannelwiseEncoder.convnet import ConvNet
from .transformerblock import TransformerBlock

class Transformer(nn.Module):
    def __init__(self,
                 enc_channels:int,
                 enc_window_size:int,
                 enc_seq_len:int,
                 enc_node_dim:int,
                 in_channels:int,
                 dim_feedforward:int,
                 n_layer:int,
                 attn_heads:int,
                 mlp_hidden_features:int,
                 mlp_class_num:int):
        super().__init__()
        
        self.encoder = ConvNet(channels=enc_channels,
                               window_size=enc_window_size,
                               seq_len=enc_seq_len,
                               node_dim=enc_node_dim,
                               mlp_mode='node_rep')
        
        self.transformer_block = TransformerBlock(in_channels=in_channels,
                                                  dim_feedforward=dim_feedforward,
                                                  n_layer=n_layer,
                                                  attn_heads=attn_heads,
                                                  mlp_hidden_features=mlp_hidden_features,
                                                  mlp_class_num=mlp_class_num)
    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer_block(x)
        return x
        
    @classmethod
    def from_config(cls, cfg):
        return cls(enc_channels=cfg['enc_channels'],
                   enc_window_size=cfg['enc_window_size'],
                   enc_seq_len=cfg['enc_seq_len'],
                   enc_node_dim=cfg['enc_node_dim'],
                   in_channels=cfg['in_channels'],
                   dim_feedforward=cfg['dim_feedforward'],
                   n_layer=cfg['n_layer'],
                   attn_heads=cfg['attn_heads'],
                   mlp_hidden_features=cfg['mlp_hidden_features'],
                   mlp_class_num=cfg['mlp_class_num'])