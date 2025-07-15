from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from .model import ECGCNN
import torch
from torch.utils.data import Dataset, DataLoader
import os

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
                print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
                print("Balanced Accuracy:", balanced_accuracy_score(all_labels, all_preds))
                del model