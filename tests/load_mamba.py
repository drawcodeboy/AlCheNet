import sys, os
sys.path.append(os.getcwd())

import torch
from torch import nn

from models import load_model

def main():
    cfg = {
        'name': 'Mamba',
        'd_model': 128,
        'n_layer': 6,
        'd_state': 16,
        'expand': 2,
        'dt_rank': 'auto',
        'd_conv': 4,
        'conv_bias': True,
        'bias': False,
        'out_dim': 128
    }
    model = load_model(cfg)
    
    p_sum = 0
    for param in model.parameters():
        p_sum += param.numel()
    print(f"Mamba Parameters: {p_sum}")
    
if __name__ == '__main__':
    main()