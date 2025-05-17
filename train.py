from datasets import load_dataset
from models import load_model

from utils import train_one_epoch, validate, save_model_ckpt, save_loss_ckpt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import argparse, time, os, sys, yaml

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
    train_data_cfg = cfg['train_data']
    train_ds = load_dataset(train_data_cfg)
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           shuffle=True,
                                           batch_size=hp_cfg['batch_size'],
                                           drop_last=True)
    print(f"Load Dataset {train_data_cfg['name']}")
    
    val_data_cfg = cfg['val_data']
    val_ds = load_dataset(val_data_cfg)
    val_dl = torch.utils.data.DataLoader(val_ds,
                                         shuffle=False,
                                         batch_size=hp_cfg['batch_size'],
                                         drop_last=False)
    print(f"Load Dataset {val_data_cfg['name']}")
            
    # Load Model
    model_cfg = cfg['model']
    print(model_cfg['name'])
    model = load_model(model_cfg).to(device)
    
    if cfg['parallel'] == True:
        model = nn.DataParallel(model)
    
    # Loss Function
    if hp_cfg['loss_fn'] == 'cross-entropy':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise Exception(f"Check loss function in configuration file")
    
    # Optimizer
    optimizer = None
    if hp_cfg['optim'] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=hp_cfg['lr'], weight_decay=0.9)
    
    # Load Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=5,
                                                     min_lr=1e-6)
    
    task = cfg['task']
    
    # Training loss
    total_train_loss = []
    total_start_time = int(time.time())
    
    min_loss = 1e4
    
    for current_epoch in range(1, hp_cfg['epochs']+1):
        print("=======================================================")
        print(f"Epoch: [{current_epoch:03d}/{hp_cfg['epochs']:03d}]\n")
        
        # Training One Epoch
        start_time = int(time.time())
        train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer, scheduler, task, device)
        elapsed_time = int(time.time() - start_time)
        print(f"Train Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s\n")

        val_loss = validate(model, val_dl, loss_fn, scheduler, task, device) # input args

        if val_loss < min_loss:
            min_loss = val_loss
            save_model_ckpt(model, cfg['save_name'], current_epoch, cfg['save_dir'])

        # total_train_loss.append(train_loss)
        # save_loss_ckpt(cfg['save_name'], total_train_loss, cfg['save_dir'])

    total_elapsed_time = int(time.time()) - total_start_time
    print(f"<Total Train Time: {total_elapsed_time//60:02d}m {total_elapsed_time%60:02d}s>")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training', parents=[add_args_parser()])
    args = parser.parse_args()

    with open(f'configs/train/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg)