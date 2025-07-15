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
    print('Files in ./data:', os.listdir('./'))
    # 1. Daten laden und vorverarbeiten
    data_dir = "./AMLS_Exercise/1.2&3_tuning_and_augmentation/data"
    X_train_split, X_val_split, y_train_split, y_val_split = load_split_data("./AMLS_Exercise/1.2&3_tuning_and_augmentation/data/split_data.pkl")
    base_dir = "./AMLS_Exercise/1.2&3_tuning_and_augmentation"
    X_train_split = pad_sequences(X_train_split)
    X_val_split = pad_sequences(X_val_split)

    augmentations = [
        lambda x: time_stretch(x, rate=np.random.uniform(0.9, 1.1)),
        lambda x: time_shift(x, shift_max=0.1),
        lambda x: add_noise(x, noise_level=0.02),
        lambda x: amplitude_scale(x, scale_range=(0.8, 1.2)),
        #lambda x: frequency_domain_augment(x, noise_level=0.01)
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
    epochs = 100  # <= 10 epochs as requested

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_counts = torch.tensor([3638, 549, 1765, 227], dtype=torch.float)
    class_weights = 1. / class_counts
    weights_tensor = class_weights.to(device)

    # Training
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
        save_path=os.path.join(base_dir, "models/pipeline_models/best_augmented.pth")
    )
    '''

    from src.inference import run_inference

    run_inference(
        model_path=base_dir+"/models/pipeline_models/best_lr0.001_wd0.0001_adam_ep100.pth",  # use your new model
        test_zip_path=data_dir+"/X_test.zip",
        output_path=data_dir+"/augment.csv",  # or another output file
        batch_size=32
    )
    # Validation
    run_validation(
        model_dir=os.path.join(base_dir, "models/pipeline_models"),
        val_data_path=os.path.join(data_dir, "val_data.pt"),
    )

if __name__ == "__main__":
    main() 