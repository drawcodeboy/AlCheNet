import torch
from torch import nn

from .rms_norm import RMSNorm
from .mamba_block import MambaBlock

class ResidualBlock(nn.Module):
    def __init__(self,
                 d_model,
                 d_inner,
                 bias,
                 conv_bias,
                 d_conv,
                 dt_rank,
                 d_state):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.mixer = MambaBlock(d_model,
                                d_inner,
                                bias,
                                conv_bias,
                                d_conv,
                                dt_rank,
                                d_state)
        self.norm = RMSNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output