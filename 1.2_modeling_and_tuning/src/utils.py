import torch

def stft_transform(X_batch, lengths=None, n_fft=256, hop_length=128):
    """
    Wendet STFT auf einen Batch von gepaddeten Zeitreihen an.
    Args:
        X_batch: Tensor (B, seq_len)
        lengths: (optional) Tensor/List mit den echten Längen (ohne Padding)
    Returns:
        Tensor (B, 1, freq_bins, time_steps)
    """
    batch_stft = []
    window = torch.hann_window(n_fft).to(X_batch.device)
    for x in X_batch:
        stft = torch.stft(
            x, n_fft=n_fft, hop_length=hop_length,
            window=window, return_complex=True
        )
        stft_mag = torch.abs(stft)
        batch_stft.append(stft_mag)
    batch_tensor = torch.stack(batch_stft).unsqueeze(1)  # (B, 1, F, T)
    return batch_tensor