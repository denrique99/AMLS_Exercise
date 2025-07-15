import os, zipfile, json, torch
from dataset import ECGDataset   # adjust to your module names
from model   import Net          # adjust to your model class

def zipped_size(path):
    return os.path.getsize(path)

def evaluate(loader, model, device="cuda"):
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred   = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()
    return correct / total

if __name__ == "__main__":
    zip_path = "../data/train.zip"
    print("→ zipped bytes:", zipped_size(zip_path))

    ds_val   = ECGDataset(split="val")           # your split logic
    val_acc  = evaluate(torch.utils.data.DataLoader(ds_val, 128, shuffle=False),
                        Net().eval())
    print("→ val-accuracy:", val_acc)
