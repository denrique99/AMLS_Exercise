import torch
from torch.utils.data import Dataset, DataLoader

class ECGSpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.unsqueeze(1)  # (N, 1, Freq, Time)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_spectrogram_dataloaders(X_train_stft, y_train, X_val_stft, y_val, batch_size=32):
    train_dataset = ECGSpectrogramDataset(X_train_stft, y_train)
    val_dataset = ECGSpectrogramDataset(X_val_stft, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader