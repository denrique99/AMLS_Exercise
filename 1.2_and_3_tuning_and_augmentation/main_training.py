from src.data import load_split_data, pad_sequences, apply_stft_numpy
from src.dataset import create_spectrogram_dataloaders
from src.model import ECGCNN
from src.train import train_and_eval
from src.validation import run_validation
import torch
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR / "models" / "pipeline_models"      
VAL_PATH =  THIS_DIR / "data" / "val_data.pt"

# 1. Daten laden und vorverarbeiten
X_train_split, X_val_split, y_train_split, y_val_split = load_split_data(THIS_DIR / "data" / "split_data.pkl")
X_train_split = pad_sequences(X_train_split)
X_val_split = pad_sequences(X_val_split)
X_train_stft = apply_stft_numpy(X_train_split)
X_val_stft = apply_stft_numpy(X_val_split)

# Speichern für spätere Validierung
torch.save({'X_val_stft': X_val_stft, 'y_val_split': y_val_split}, VAL_PATH)
print(f"Saved STFT validation data to {VAL_PATH}")

# 2. DataLoader bauen
train_loader, val_loader = create_spectrogram_dataloaders(X_train_stft, y_train_split, X_val_stft, y_val_split)

# 3. Modell bauen und trainieren
model = ECGCNN()

lr = 0.001
weight_decay = 0.0001
optimizer_type = "adam"
epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_counts = torch.tensor([3638, 549, 1765, 227], dtype=torch.float)
class_weights = 1. / class_counts
weights_tensor = class_weights.to(device)

# 4. Training und Mini-Evaluation

train_and_eval(
    model,
    train_loader,
    val_loader,
    lr=lr,
    weight_decay=weight_decay,
    optimizer_type=optimizer_type,
    epochs=epochs,
    weights_tensor=weights_tensor,
    device=device,
    save_path= f"{THIS_DIR}/models/pipeline_models/best_lr{lr}_wd{weight_decay}_{optimizer_type}_ep{epochs}.pth"
)

# 4. Validierung
run_validation(
        model_dir=MODEL_DIR,
        val_data_path=VAL_PATH,
    )