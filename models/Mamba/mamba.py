"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union

from .rms_norm import RMSNorm
from .residual_block import ResidualBlock

class Mamba(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_layer: int,
                 d_inner: int,
                 d_state: int = 16,
                 expand: int = 2,
                 dt_rank: Union[int, str] = 'auto',
                 d_conv: int = 4 ,
                 conv_bias: bool = True,
                 bias: bool = False,
                 out_dim: int = 128,):
        super().__init__()
        
        self.layers = nn.ModuleList([ResidualBlock(d_model,
                                                   d_inner,
                                                   bias,
                                                   conv_bias,
                                                   d_conv,
                                                   dt_rank,
                                                   d_state) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, out_dim, bias=False)

    def forward(self, x):
        """
        Args:
            x (float tensor): shape (b, l, dim)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits
    
    @classmethod
    def from_config(cls, cfg):
        if cfg['dt_rank'] == 'auto':
            dt_rank = math.ceil(cfg['d_model'] / 16)
            
        d_inner = int(cfg['expand'] * cfg['d_model'])
        
        return cls(d_model=cfg['d_model'],
                   n_layer=cfg['n_layer'],
                   d_inner=d_inner,
                   d_state=cfg['d_state'],
                   expand=cfg['expand'],
                   dt_rank=dt_rank,
                   d_conv=cfg['d_conv'],
                   conv_bias=cfg['conv_bias'],
                   bias=cfg['bias'],
                   out_dim=cfg['out_dim'])