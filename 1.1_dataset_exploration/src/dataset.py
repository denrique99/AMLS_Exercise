import torch
from torch.utils.data import Dataset, DataLoader


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

class ECGSpectrogramDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X.unsqueeze(1)  # (N, 1, Freq, Time)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            x = self.apply_augmentations(x)

        return x, y

    def apply_augmentations(self, x):
        # Random time shift
        if random.random() < 0.5:
            shift = random.randint(-10, 10)
            x = torch.roll(x, shifts=shift, dims=-1)

        # Add small Gaussian noise
        if random.random() < 0.5:
            noise = torch.randn_like(x) * 0.01
            x = x + noise

        # Random amplitude scaling
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            x = x * scale

        return x


def create_spectrogram_dataloaders(X_train_stft, y_train, X_val_stft, y_val, batch_size=32, augment=False):
    train_dataset = ECGSpectrogramDataset(X_train_stft, y_train, augment=augment)
    val_dataset = ECGSpectrogramDataset(X_val_stft, y_val, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader








# class ECGSpectrogramDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X.unsqueeze(1)  # (N, 1, Freq, Time)
#         self.y = torch.tensor(y, dtype=torch.long)
#     def __len__(self):
#         return len(self.X)
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# def create_spectrogram_dataloaders(X_train_stft, y_train, X_val_stft, y_val, batch_size=32):
#     train_dataset = ECGSpectrogramDataset(X_train_stft, y_train)
#     val_dataset = ECGSpectrogramDataset(X_val_stft, y_val)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     return train_loader, val_loader