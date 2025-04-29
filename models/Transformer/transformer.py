import torch
from torch import nn

from .pos_embedding import SinusoidalPosEmbedding
from .rms_norm import RMSNorm

class Transformer(nn.Module):
    def __init__(self,
                 in_features:int = 19,
                 d_model:int = 128,
                 n_layer:int = 2,
                 attn_heads:int = 2,
                 out_dim:int = 32,
                 device:str = 'cuda'):
        super().__init__()
        
        self.li = nn.Linear(in_features, d_model)
        self.relu = nn.ReLU()
        
        self.pos_emb = SinusoidalPosEmbedding(dim=d_model)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model,
                                       nhead=attn_heads,
                                       batch_first=True) for i in range(0, n_layer)
        ])
        
        self.norm_f = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, out_dim, bias=False)
        
        self.device = device
        
    def forward(self, x):
        B, L, D = x.size() # Batch, Sequence Length, Dimension
        
        x = self.relu(self.li(x))
        
        timesteps = torch.arange(0, L, 1)
        pos_emb = self.pos_emb(timesteps).to(self.device)
        
        x = x + pos_emb
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        return x

    @classmethod
    def from_config(cls, cfg):
        return cls(in_features=cfg['in_features'],
                   d_model=cfg['d_model'],
                   n_layer=cfg['n_layer'],
                   attn_heads=cfg['attn_heads'],
                   out_dim=cfg['out_dim'],
                   device=cfg['device'])