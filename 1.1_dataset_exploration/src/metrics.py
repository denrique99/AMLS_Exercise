from collections import Counter
import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Create confusion matrix without sklearn
    """
    # Get unique labels
    labels = sorted(list(set(y_true + y_pred)))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Initialize matrix
    matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    
    # Fill matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        matrix[true_idx][pred_idx] += 1
    
    return matrix, labels

def classification_metrics(y_true, y_pred):
    """
    Calculate precision, recall, f1-score for each class
    """
    # Get unique labels
    labels = sorted(list(set(y_true + y_pred)))
    
    metrics = {}
    
    for label in labels:
        # True positives, false positives, false negatives
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Support (number of true instances)
        support = sum(1 for true in y_true if true == label)
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
    
    return metrics, labels

def balanced_accuracy_score(y_true, y_pred):
    """
    Calculate balanced accuracy (average of recall for each class)
    """
    metrics, labels = classification_metrics(y_true, y_pred)
    
    # Calculate balanced accuracy as average of recall scores
    total_recall = sum(metrics[label]['recall'] for label in labels)
    balanced_acc = total_recall / len(labels)
    
    return balanced_acc

def classification_report(y_true, y_pred, digits=4):
    """
    Generate classification report similar to sklearn
    """
    metrics, labels = classification_metrics(y_true, y_pred)
    
    # Header
    report = f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n"
    report += "\n"
    
    # Per-class metrics
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
    
    # Averages
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

def print_metrics(y_true, y_pred):
    """
    Print classification metrics without sklearn
    """
    # Classification report
    print(classification_report(y_true, y_pred, digits=4))
    
    # Confusion matrix
    conf_matrix, labels = confusion_matrix(y_true, y_pred)
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
    
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

# Example usage:
# y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
# y_pred = [0, 1, 1, 0, 1, 2, 0, 2, 2]
# print_metrics(y_true, y_pred)