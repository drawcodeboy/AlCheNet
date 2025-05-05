import sys, os
sys.path.append(os.getcwd())

import torch
from torch import nn

from models import load_model
import yaml

def main():
    with open(f"configs/train/transformer.yaml") as f:
        cfg = yaml.full_load(f)
    
    model_cfg = cfg['model']
    model = load_model(model_cfg)
    
    p_sum = 0
    for name, p in model.named_parameters():
        p_sum += p.numel()
        print(f"{name:30s}--{p.numel()}")
    print(f"Transformer: {p_sum}")
    
    with open(f"configs/train/graphnet.yaml") as f:
        cfg = yaml.full_load(f)
    
    model_cfg = cfg['model']
    model = load_model(model_cfg)
    
    p_sum = 0
    for p in model.parameters():
        p_sum += p.numel()
    print(f"GraphNet: {p_sum}")
    
if __name__ == '__main__':
    main()