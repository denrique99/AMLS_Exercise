from src.data import load_split_data, pad_sequences, apply_stft_numpy
from src.dataset import create_spectrogram_dataloaders
from src.model import ECGCNN
from src.train import train_and_eval
from src.validation import run_validation

# 1. Daten laden und vorverarbeiten
X_train_split, X_val_split, y_train_split, y_val_split = load_split_data("data/split_data.pkl")
X_train_split = pad_sequences(X_train_split)
X_val_split = pad_sequences(X_val_split)
X_train_stft = apply_stft_numpy(X_train_split)
X_val_stft = apply_stft_numpy(X_val_split)

# 2. DataLoader bauen
train_loader, val_loader = create_spectrogram_dataloaders(X_train_stft, y_train_split, X_val_stft, y_val_split)

# 3. Modell bauen und trainieren
model = ECGCNN()

lr = 0.001
weight_decay = 0.0001
optimizer_type = "adam"
epochs = 10

# 4. Training und Mini-Evaluation
train_and_eval(
    model,
    train_loader,
    val_loader,
    lr=lr,
    weight_decay=weight_decay,
    optimizer_type=optimizer_type,
    epochs=epochs,
    save_path= f"models/pipeline_models/best_lr{lr}_wd{weight_decay}_{optimizer_type}_ep{epochs}.pth"
)

# 4. Validierung
run_validation(
        model_dir=f"models/pipeline_models",
        val_data_path="data/val_data.pt",
    )