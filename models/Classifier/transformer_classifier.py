import torch
from torch import nn

from ..Transformer.transformer import Transformer
from .mlp import MLP

class TransformerClassifier(nn.Module):
    def __init__(self,
                 in_features:int = 19,
                 d_model:int = 128,
                 n_layer:int = 2,
                 attn_heads:int = 2,
                 out_dim:int = 32,
                 mid_features:int = 50,
                 class_num:int = 3,
                 device:str = 'cuda'):
        super().__init__()
        
        self.transformer = Transformer(in_features=in_features,
                                       d_model=d_model,
                                       n_layer=n_layer,
                                       attn_heads=attn_heads,
                                       out_dim=out_dim,
                                       device=device)
        self.mlp = MLP(in_features=out_dim,
                       mid_features=mid_features,
                       class_num=class_num)
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.mlp(x)
        
        return x
        
    @classmethod
    def from_config(cls, cfg):
        return cls(in_features=cfg['in_features'],
                   d_model=cfg['d_model'],
                   n_layer=cfg['n_layer'],
                   attn_heads=cfg['attn_heads'],
                   out_dim=cfg['out_dim'],
                   mid_features=cfg['mid_features'],
                   class_num=cfg['class_num'],
                   device=cfg['device'])