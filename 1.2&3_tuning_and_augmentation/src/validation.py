from .model import ECGCNN
import torch
from torch.utils.data import Dataset, DataLoader
import os

# Import our manual metrics functions (assuming they're in the same module or imported)
from collections import Counter
import numpy as np

def confusion_matrix(y_true, y_pred):
    """Create confusion matrix without sklearn"""
    labels = sorted(list(set(y_true + y_pred)))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        matrix[true_idx][pred_idx] += 1
    
    return matrix, labels

def classification_metrics(y_true, y_pred):
    """Calculate precision, recall, f1-score for each class"""
    labels = sorted(list(set(y_true + y_pred)))
    
    metrics = {}
    
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        support = sum(1 for true in y_true if true == label)
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
    
    return metrics, labels

def balanced_accuracy_score(y_true, y_pred):
    """Calculate balanced accuracy (average of recall for each class)"""
    metrics, labels = classification_metrics(y_true, y_pred)
    
    total_recall = sum(metrics[label]['recall'] for label in labels)
    balanced_acc = total_recall / len(labels)
    
    return balanced_acc

def classification_report(y_true, y_pred, digits=4):
    """Generate classification report similar to sklearn"""
    metrics, labels = classification_metrics(y_true, y_pred)
    
    report = f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n"
    report += "\n"
    
    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    
    for label in labels:
        m = metrics[label]
        report += f"{str(label):>12} {m['precision']:>10.{digits}f} {m['recall']:>10.{digits}f} {m['f1-score']:>10.{digits}f} {m['support']:>10}\n"
        
        total_support += m['support']
        weighted_precision += m['precision'] * m['support']
        weighted_recall += m['recall'] * m['support']
        weighted_f1 += m['f1-score'] * m['support']
    
    report += "\n"
    
    # Macro average
    macro_precision = sum(metrics[label]['precision'] for label in labels) / len(labels)
    macro_recall = sum(metrics[label]['recall'] for label in labels) / len(labels)
    macro_f1 = sum(metrics[label]['f1-score'] for label in labels) / len(labels)
    
    report += f"{'macro avg':>12} {macro_precision:>10.{digits}f} {macro_recall:>10.{digits}f} {macro_f1:>10.{digits}f} {total_support:>10}\n"
    
    # Weighted average
    if total_support > 0:
        weighted_precision /= total_support
        weighted_recall /= total_support
        weighted_f1 /= total_support
    
    report += f"{'weighted avg':>12} {weighted_precision:>10.{digits}f} {weighted_recall:>10.{digits}f} {weighted_f1:>10.{digits}f} {total_support:>10}\n"
    
    return report

class ECGSpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_val_loader(val_data_path, batch_size=32):
    data = torch.load(val_data_path)
    X_val_stft = data['X_val_stft']
    y_val_split = data['y_val_split']
    val_dataset = ECGSpectrogramDataset(X_val_stft, y_val_split)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader

def run_validation(model_dir, val_data_path, batch_size=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_loader = get_val_loader(val_data_path, batch_size)
    
    for root, _, files in os.walk(model_dir):
        for filename in files:
            if filename.endswith(".pth"):
                full_path = os.path.join(root, filename)
                print(f"Verarbeite Modell: {full_path}")
                
                all_preds = []
                all_labels = []
                
                model = ECGCNN()
                model.load_state_dict(torch.load(full_path, map_location=device))
                model.to(device)
                model.eval()
                
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        X_val = X_val.to(device)
                        outputs = model(X_val)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(y_val.numpy())
                
                print(f"\nModel: {filename}")
                print(classification_report(all_labels, all_preds, digits=4))
                
                # Print confusion matrix
                conf_matrix, labels = confusion_matrix(all_labels, all_preds)
                print("Confusion Matrix:")
                
                # Print header
                print(f"{'':>8}", end="")
                for label in labels:
                    print(f"{str(label):>8}", end="")
                print()
                
                # Print matrix with row labels
                for i, label in enumerate(labels):
                    print(f"{str(label):>8}", end="")
                    for j in range(len(labels)):
                        print(f"{conf_matrix[i][j]:>8}", end="")
                    print()
                
                balanced_acc = balanced_accuracy_score(all_labels, all_preds)
                print(f"Balanced Accuracy: {balanced_acc:.4f}")
                
                del model