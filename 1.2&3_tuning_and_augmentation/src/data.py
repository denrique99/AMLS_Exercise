import pickle
import numpy as np
import torch

def load_split_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def pad_sequences(sequences, max_len=None):
    if not max_len:
        max_len = max(len(seq) for seq in sequences)
    return np.array([np.pad(seq, (0, max_len - len(seq))) for seq in sequences])

def apply_stft_numpy(X, n_fft=256, hop_length=128):
    data_tensor = torch.tensor(X, dtype=torch.float32)
    window = torch.hann_window(n_fft)
    stft_results = []
    for i in range(data_tensor.shape[0]):
        stft = torch.stft(
            data_tensor[i], n_fft=n_fft, hop_length=hop_length,
            window=window, return_complex=True
        )
        stft_magnitude = torch.abs(stft)
        stft_results.append(stft_magnitude)
    stft_tensor = torch.stack(stft_results)
    return stft_tensor