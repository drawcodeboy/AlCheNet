import torch
from torch import nn
from typing import Union
import math

from ..Mamba.mamba import Mamba
from .mlp import MLP

class MambaClassifier(nn.Module):
    def __init__(self,
                 in_features: int,
                 d_model: int,
                 n_layer: int,
                 d_inner: int,
                 d_state: int = 16,
                 expand: int = 2,
                 dt_rank: Union[int, str] = 'auto',
                 d_conv: int = 4 ,
                 conv_bias: bool = True,
                 bias: bool = False,
                 out_dim: int = 128,
                 mid_features: int = 50,
                 class_num: int = 3,
                 device: str = 'cuda'):
        super().__init__()
        
        self.mamba = Mamba(in_features=in_features,
                           d_model=d_model,
                           n_layer=n_layer,
                           d_inner=d_inner,
                           d_state=d_state,
                           expand=expand,
                           dt_rank=dt_rank,
                           d_conv=d_conv,
                           conv_bias=conv_bias,
                           bias=bias,
                           out_dim=out_dim,
                           device=device)
        self.mlp = MLP(in_features=out_dim,
                       mid_features=mid_features,
                       class_num=class_num)
    
    def forward(self, x):
        x = self.mamba(x)
        x = self.mlp(x)
        
        return x
        
    @classmethod
    def from_config(cls, cfg):
        if cfg['dt_rank'] == 'auto':
            dt_rank = math.ceil(cfg['d_model'] / 16)
            
        d_inner = int(cfg['expand'] * cfg['d_model'])
        
        return cls(in_features=cfg['in_features'],
                   d_model=cfg['d_model'],
                   n_layer=cfg['n_layer'],
                   d_inner=d_inner,
                   d_state=cfg['d_state'],
                   expand=cfg['expand'],
                   dt_rank=dt_rank,
                   d_conv=cfg['d_conv'],
                   conv_bias=cfg['conv_bias'],
                   bias=cfg['bias'],
                   out_dim=cfg['out_dim'],
                   mid_features=cfg['mid_features'],
                   class_num=cfg['class_num'],
                   device=cfg['device'])