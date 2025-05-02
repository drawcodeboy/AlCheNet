import torch
from torch import nn

from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    
class MLP(nn.Module):
    def __init__(self,
                 in_features: int,
                 mid_features: int,
                 class_num: int):
        super().__init__()
        
        self.norm1 = RMSNorm(d_model=in_features)
        self.l1 = nn.Linear(in_features=in_features, out_features=mid_features)
        self.norm2 = RMSNorm(d_model=mid_features)
        self.l2 = nn.Linear(in_features=mid_features, out_features=class_num)
        
    def forward(self, x):
        # x shape: (B, L, D)
        
        x = self.norm1(x)

        # Global Average Pooling
        # print(x.shape)
        x = torch.mean(x, dim=1)

        # MLP
        x = self.l1(x)

        x = self.l2(self.norm2(x))
        
        return x