import torch

import argparse
import time, sys, os, yaml

from utils import evaluate
from models import load_model
from datasets import load_dataset

def add_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str)
    
    return parser

def main(cfg):
    print(f"=====================[{cfg['expr']}]=====================")
    
    # Device Setting
    device = None
    if cfg['device'] != 'cpu' and torch.cuda.is_available():
        device = cfg['device']
    else: 
        device = 'cpu'
    print(f"device: {device}")
    
    # Hyperparameter Settings
    hp_cfg = cfg['hyperparameters']
    
    # Load Dataset
    data_cfg = cfg['data']
    test_ds = load_dataset(data_cfg)
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=hp_cfg['batch_size'])
    print(f"Load Dataset {data_cfg['dataset']}")
    
    # Load Model
    model_cfg = cfg['model']
    model = load_model(model_cfg).to(device)
    ckpt = torch.load(os.path.join(cfg['save_dir'], cfg['weights_file_name']),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    start_time = int(time.time())
    result = evaluate(model, test_dl, device)
    test_time = int(time.time() - start_time)
    print(f"Test Time: {test_time//60:02d}m {test_time%60:02d}s")
    
    for key, value in result.items():
        print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test', parents=[add_args_parser()])
    args = parser.parse_args()

    with open(f'configs/test/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg)