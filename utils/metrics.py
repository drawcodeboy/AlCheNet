import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_metrics(outputs, targets, class_num:int=10):
    results = {
        "Accuracy": 0.,
        "F1-Score(Macro)":0.,
        "Precision(Macro)":0.,
        "Recall(Macro)":0.,
    }
    
    accuracy = accuracy_score(targets, outputs)
    macro_f1 = f1_score(targets, outputs, average='macro')
    precision = precision_score(targets, outputs, average='macro')
    recall = recall_score(targets, outputs, average='macro')
    
    results['Accuracy'] = accuracy
    results["F1-Score(Macro)"] = macro_f1
    results["Precision(Macro)"] = precision
    results["Recall(Macro)"] = recall
    
    return results