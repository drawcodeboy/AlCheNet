import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def multiclass_sensitivity_specificity(y_true, y_pred, average='macro'):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    sensitivities = []
    specificities = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    if average == 'macro':
        return {
            'sensitivity': np.mean(sensitivities),
            'specificity': np.mean(specificities)
        }
    elif average == 'none':
        return {
            'sensitivity': sensitivities,
            'specificity': specificities
        }
    else:
        raise ValueError("average must be 'macro' or 'none'")

def get_metrics(outputs, targets, class_num:int=10):
    results = {
        "Accuracy": 0.,
        "F1-Score(Macro)":0.,
        "Precision(Macro)":0.,
        "Recall(Macro)":0.,
        "Sensitivity":0.,
        "Specificity":0.
    }
    
    accuracy = accuracy_score(targets, outputs)
    macro_f1 = f1_score(targets, outputs, average='macro')
    precision = precision_score(targets, outputs, average='macro')
    recall = recall_score(targets, outputs, average='macro')
    
    res_dict = multiclass_sensitivity_specificity(targets, outputs)
    
    results['Accuracy'] = accuracy
    results["F1-Score(Macro)"] = macro_f1
    results["Precision(Macro)"] = precision
    results["Recall(Macro)"] = recall
    results["Sensitivity"] = res_dict['sensitivity']
    results["Specificity"] = res_dict['specificity']
    
    return results