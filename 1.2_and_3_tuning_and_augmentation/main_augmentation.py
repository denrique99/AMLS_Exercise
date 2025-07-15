import os
import numpy as np
from src.augmentation import (
    time_stretch, time_shift, add_noise, random_crop,
    resample_signal, amplitude_scale, apply_augmentations, ensure_length
)
from src.data import load_split_data, pad_sequences, apply_stft_numpy
from src.dataset import create_spectrogram_dataloaders
from src.model import ECGCNN
from src.train import train_and_eval
from src.validation import run_validation
import torch


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # z.B. .../1.2_and_3_tuning_and_augmentation
    data_dir = os.path.join(base_dir, "data")
    
    print('Files in data_dir:', os.listdir(data_dir))

    # 1. Daten laden und vorverarbeiten
    split_path = os.path.join(data_dir, "split_data.pkl")
    X_train_split, X_val_split, y_train_split, y_val_split = load_split_data(split_path)

    X_train_split = pad_sequences(X_train_split)
    X_val_split = pad_sequences(X_val_split)

    augmentations = [
        lambda x: time_stretch(x, rate=np.random.uniform(0.9, 1.1)),
        lambda x: time_shift(x, shift_max=0.1),
        lambda x: add_noise(x, noise_level=0.02),
        lambda x: amplitude_scale(x, scale_range=(0.8, 1.2)),
    ]

    original_length = X_train_split[0].shape[0]
    X_train_augmented = [
        ensure_length(apply_augmentations(x, augmentations), original_length)
        for x in X_train_split
    ]
    X_train_combined = np.stack([*X_train_split, *X_train_augmented])
    y_train_combined = np.concatenate([y_train_split, y_train_split])

    X_train_stft = apply_stft_numpy(X_train_combined)
    X_val_stft = apply_stft_numpy(X_val_split)

    train_loader, val_loader = create_spectrogram_dataloaders(
        X_train_stft, y_train_combined, X_val_stft, y_val_split
    )

    # Model, optimizer, and training settings
    model = ECGCNN()
    lr = 0.001
    weight_decay = 0.0001
    optimizer_type = "adam"
    epochs = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_counts = torch.tensor([3638, 549, 1765, 227], dtype=torch.float)
    class_weights = 1. / class_counts
    weights_tensor = class_weights.to(device)

    # Training (optional, wenn nicht auskommentiert)
    '''
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
        save_path=os.path.join(base_dir, "models", "pipeline_models", "best_augmented.pth")
    )
    '''

    from src.inference import run_inference

    run_inference(
        model_path=os.path.join(base_dir, "models", "pipeline_models", "best_lr0.001_wd0.0001_adam_ep100.pth"),
        test_zip_path="data/X_test.zip",
        output_path="augment.csv",
        batch_size=32
    )

    from src.validation import run_validation

    run_validation(
        model_dir=os.path.join(base_dir, "models", "pipeline_models"),
        val_data_path=os.path.join(data_dir, "val_data.pt"),
    )

if __name__ == "__main__":
    main()  