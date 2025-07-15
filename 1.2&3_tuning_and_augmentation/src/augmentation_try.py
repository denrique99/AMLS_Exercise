from augmentation import (
    time_stretch, time_shift, add_noise, random_crop,
    resample_signal, amplitude_scale, apply_augmentations, ensure_length
)
from data import load_split_data, pad_sequences, apply_stft_numpy
from dataset import create_spectrogram_dataloaders

import torch

# 1. Daten laden und vorverarbeiten
X_train_split, X_val_split, y_train_split, y_val_split = load_split_data("data/split_data.pkl")
X_train_split = pad_sequences(X_train_split)
X_val_split = pad_sequences(X_val_split)
X_train_stft = apply_stft_numpy(X_train_split)
X_val_stft = apply_stft_numpy(X_val_split)

import numpy as np

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

try:
    X_train_combined = np.stack([*X_train_split, *X_train_augmented])
except ValueError as e:
    print("Shape mismatch! Printing shapes for debugging:")
    print("Original:", [x.shape for x in X_train_split])
    print("Augmented:", [x.shape for x in X_train_augmented])
    raise e

y_train_combined = np.concatenate([y_train_split, y_train_split])

X_train_stft = apply_stft_numpy(X_train_combined)
train_loader, val_loader = create_spectrogram_dataloaders(
    X_train_stft, y_train_combined, X_val_stft, y_val_split
)

from src.inference import run_inference

run_inference(
    model_path="models/pipeline_models/best_augmented.pth",  # use your new model
    test_zip_path="data/X_test.zip",
    output_path="data/augment.csv",  # or another output file
    batch_size=32
)