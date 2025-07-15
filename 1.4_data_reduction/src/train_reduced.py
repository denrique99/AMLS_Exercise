import argparse
import pickle
import json
import torch
import torch.nn.functional as F
from model import ECGCNN
from dataset import create_spectrogram_dataloaders


def apply_stft(signals, n_fft=64, hop_length=16, pad_to=None):
    stft_tensors, max_t = [], 0
    for signal in signals:
        sig = torch.tensor(signal, dtype=torch.float32)
        m = torch.stft(
            sig,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True
        ).abs()
        stft_tensors.append(m)
        max_t = max(max_t, m.shape[1])
    # Determine padding length
    if pad_to is not None:
        pad_len = pad_to
    else:
        pad_len = max_t
        if pad_len % 4 != 0:
            pad_len = ((pad_len + 3) // 4) * 4
    padded = []
    for m in stft_tensors:
        pad = pad_len - m.shape[1]
        padded.append(F.pad(m, (0, pad)))
    return torch.stack(padded), pad_len

def train_on_subset(X_train_stft, y_train, X_val_stft, y_val, title=""):
    train_loader, val_loader = create_spectrogram_dataloaders(
        X_train_stft, y_train, X_val_stft, y_val, batch_size=32)
    model = ECGCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"{title} - Val Accuracy: {acc:.4f}")
    torch.save(model.state_dict(), "reduced_model.pth")
    print("Model saved as reduced_model.pth")
    return model, acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_data", type=str, default="../data/split_data.pkl")
    parser.add_argument("--indices", type=str, default="reduced_indices.json")
    args = parser.parse_args()

    # Load full split data
    with open(args.split_data, "rb") as f:
        X_train_split, X_val_split, y_train_split, y_val_split = pickle.load(f)

    # Load reduced indices
    with open(args.indices, "r") as f:
        idxs = json.load(f)

    # Subset the data
    X_train_reduced = [X_train_split[i] for i in idxs]
    y_train_reduced = [y_train_split[i] for i in idxs]

    # Apply STFT to both sets, using the same pad_len
    X_train_stft, pad_len_train = apply_stft(X_train_reduced)
    X_val_stft, pad_len_val = apply_stft(X_val_split)
    pad_len = max(pad_len_train, pad_len_val)
    X_train_stft, _ = apply_stft(X_train_reduced, pad_to=pad_len)
    X_val_stft, _ = apply_stft(X_val_split, pad_to=pad_len)

    # Train and evaluate
    train_on_subset(X_train_stft, y_train_reduced, X_val_stft, y_val_split, title="Reduced Training Set")

if __name__ == "__main__":
    main() 