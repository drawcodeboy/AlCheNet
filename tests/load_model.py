import sys, os
sys.path.append(os.getcwd())

import torch
from torch import nn

from models import load_model

def main():
    cfg = {
        'name': 'Mamba',
        'in_features': 19,
        'd_model': 128,
        'n_layer': 6,
        'd_state': 16,
        'expand': 2,
        'dt_rank': 'auto',
        'd_conv': 4,
        'conv_bias': True,
        'bias': False,
        'out_dim': 128,
        'device': 'cuda'
    }
    model = load_model(cfg).to(cfg['device'])
    
    p_sum = 0
    for param in model.parameters():
        p_sum += param.numel()
    print(f"Mamba Parameters: {p_sum * 10e-7:.2f}M")
    
    input_t = torch.randn(16, 500, 19).to(cfg['device'])
    output_t = model(input_t)
    print(f"Output Shape: {output_t.shape}")
    
    cfg = {
        'name': 'Transformer',
        'in_features': 19,
        'd_model': 128,
        'n_layer': 6,
        'attn_heads': 1,
        'out_dim': 128,
        'device': 'cuda'
    }
    model = load_model(cfg).to(cfg['device'])
    
    p_sum = 0
    for param in model.parameters():
        p_sum += param.numel()
    print(f"Transformer Parameters: {p_sum * 10e-7:.2f}M")
    
    input_t = torch.randn(16, 500, 19).to(cfg['device'])
    output_t = model(input_t)
    print(f"Output Shape: {output_t.shape}")
    
if __name__ == '__main__':
    main()