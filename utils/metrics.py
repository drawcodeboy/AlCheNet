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
    
def compute_classwise_and_macro_metrics(y_true, y_pred, num_classes):
    per_class_metrics = {
        "class": [],
        "precision": [],
        "recall": [],
        "f1-score": []
    }

    for i in range(num_classes):
        # 해당 클래스만 양성으로 간주
        y_true_bin = (np.array(y_true) == i).astype(int)
        y_pred_bin = (np.array(y_pred) == i).astype(int)

        precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        per_class_metrics["class"].append(i)
        per_class_metrics["precision"].append(precision)
        per_class_metrics["recall"].append(recall)
        per_class_metrics["f1-score"].append(f1)

    # Macro 평균
    macro_metrics = {
        "macro_precision": np.mean(per_class_metrics["precision"]),
        "macro_recall": np.mean(per_class_metrics["recall"]),
        "macro_f1": np.mean(per_class_metrics["f1-score"])
    }

    return per_class_metrics, macro_metrics

def get_metrics(outputs, targets, class_num:int=10):
    results = {
        "Accuracy": 0.,
        "F1-Score(Macro)":0.,
        "Precision(Macro)":0.,
        "Recall(Macro)":0.,
        "Sensitivity":0.,
        "Specificity":0.
    }
    
    average = 'macro'
    accuracy = accuracy_score(targets, outputs)
    macro_f1 = f1_score(targets, outputs, average=average)
    precision = precision_score(targets, outputs, average=average)
    recall = recall_score(targets, outputs, average=average)
    
    res_dict = multiclass_sensitivity_specificity(targets, outputs)
    
    results['Accuracy'] = accuracy
    results["F1-Score(Macro)"] = macro_f1
    results["Precision(Macro)"] = precision
    results["Recall(Macro)"] = recall
    results["Sensitivity"] = res_dict['sensitivity']
    results["Specificity"] = res_dict['specificity']
    
    per_class, macro = compute_classwise_and_macro_metrics(targets, outputs, num_classes=3)

    print("Class-wise metrics:")
    for i in range(len(per_class["class"])):
        print(f"Class {per_class['class'][i]} | "
            f"Precision: {per_class['precision'][i]:.2f} | "
            f"Recall: {per_class['recall'][i]:.2f} | "
            f"F1: {per_class['f1-score'][i]:.2f}")

    print("Macro Averages:")
    for k, v in macro.items():
        print(f"{k}: {v:.4f}")
        
    print("Accuracy")
    print(f"Acc: {accuracy:.4f}")
    
    return results