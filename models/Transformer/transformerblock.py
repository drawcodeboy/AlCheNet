import torch
from torch import nn

from ..mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 dim_feedforward:int,
                 n_layer:int,
                 attn_heads:int,
                 mlp_hidden_features:int,
                 mlp_class_num:int):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=in_channels,
                                       nhead=attn_heads,
                                       dim_feedforward=dim_feedforward,
                                       batch_first=True) for i in range(0, n_layer)
        ])
        
        self.mlp = MLP(in_features=in_channels,
                       hidden_features=mlp_hidden_features,
                       class_num=mlp_class_num)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_channels))
        
    def forward(self, x):
        B = x.size(0)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1) 
        
        for layer in self.layers:
            x = layer(x)
            
        # x = x.mean(dim = 1)
        
        x = x[:, 0] # Class Token
        x = self.mlp(x)
                
        return x