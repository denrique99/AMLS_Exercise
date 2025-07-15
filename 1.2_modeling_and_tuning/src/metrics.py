from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

def print_metrics(y_true, y_pred):
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))