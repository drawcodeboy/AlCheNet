import torch
import torch.nn.functional as F

import numpy as np

from .metrics import get_metrics

def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, task, device):
    model.train()
    total_loss = []
    
    for batch_idx, (x, target) in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        
        if task == 'freq':
            x = x['freq']
            x = x.to(device)
        else:
            x['freq'] = x['freq'].to(device)
            x['edge_index'] = x['edge_index'].to(device)
            x['edge_weight'] = x['edge_weight'].to(device)
            
        target = target.to(device)
        
        logits = model(x)
        # print(logits.shape, target.shape)
        loss = loss_fn(logits, target)
        
        total_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, Loss: {sum(total_loss)/len(total_loss):.6f}, LR: {scheduler.get_last_lr()[0]:.6f}", end="")
    print()
    
    return sum(total_loss)/len(total_loss)

@torch.no_grad()
def validate(model, dataloader, loss_fn, scheduler, task, device):
    model.eval()
    total_loss = []
    
    for batch_idx, (x, target) in enumerate(dataloader, start=1):
        if task == 'freq':
            x = x['freq']
            x = x.to(device)
        else:
            x['freq'] = x['freq'].to(device)
            x['edge_index'] = x['edge_index'].to(device)
            x['edge_weight'] = x['edge_weight'].to(device)
        target = target.to(device)
        
        logits = model(x)
        loss = loss_fn(logits, target)
        
        total_loss.append(loss.item())
                
        print(f"\rValidate: {100*batch_idx/len(dataloader):.2f}%, Loss: {sum(total_loss)/len(total_loss):.6f}", end="")
    print()
    
    scheduler.step(sum(total_loss)/len(total_loss))
    
    return sum(total_loss)/len(total_loss)

@torch.no_grad()
def evaluate(model, dataloader, task, device):
    model.eval()
    
    total_outputs = []
    total_targets = []
    
    for batch_idx, (x, target) in enumerate(dataloader, start=1):
        if task == 'freq':
            x = x['freq']
            x = x.to(device)
        else:
            x['freq'] = x['freq'].to(device)
            x['edge_index'] = x['edge_index'].to(device)
            x['edge_weight'] = x['edge_weight'].to(device)
        x = x.to(device)
        target = target.to(device)
        
        logits = model(x)
        out = F.softmax(logits, dim=1)
        out = torch.argmax(out, dim=1)
        
        total_outputs.extend(out.tolist())
        total_targets.extend(target.tolist())
        
        print(f"\rEvaluate: {100*batch_idx/len(dataloader):.2f}%", end="")
    print()
    
    result = get_metrics(np.array(total_outputs), np.array(total_targets))
    
    return result