import numpy as np
from scipy.signal import resample
from scipy.fft import fft, ifft


def time_stretch(ecg, rate=1.1, target_length=None):
    """Stretches or compresses the ECG signal in time by the given rate."""
    n = int(len(ecg) / rate)
    stretched = resample(ecg, n)
    if target_length is not None:
        return ensure_length(stretched, target_length)
    return stretched


def time_shift(ecg, shift_max=0.1):
    """Shifts the ECG signal in time by a random fraction of its length."""
    shift = int(np.random.uniform(-shift_max, shift_max) * len(ecg))
    return np.roll(ecg, shift)


def add_noise(ecg, noise_level=0.01):
    """Adds Gaussian noise to the ECG signal."""
    noise = np.random.normal(0, noise_level, size=ecg.shape)
    return ecg + noise


def random_crop(ecg, crop_size=0.9):
    """Randomly crops the ECG signal to a fraction of its original length and pads to original size."""
    n = int(len(ecg) * crop_size)
    start = np.random.randint(0, len(ecg) - n)
    cropped = ecg[start:start + n]
    pad_left = start
    pad_right = len(ecg) - n - pad_left
    return np.pad(cropped, (pad_left, pad_right), mode='constant')


def resample_signal(ecg, new_length):
    """Resamples the ECG signal to a new length."""
    return resample(ecg, new_length)


def amplitude_scale(ecg, scale_range=(0.8, 1.2)):
    """Scales the amplitude of the ECG signal by a random factor."""
    scale = np.random.uniform(*scale_range)
    return ecg * scale


def ensure_length(ecg, target_length):
    if len(ecg) == target_length:
        return ecg
    elif len(ecg) > target_length:
        # Crop
        return ecg[:target_length]
    else:
        # Pad
        return np.pad(ecg, (0, target_length - len(ecg)), mode='constant')


def apply_augmentations(ecg, augmentations=None):
    """Apply a list of augmentation functions to the ECG signal in sequence."""
    if augmentations is None:
        augmentations = []
    for aug in augmentations:
        ecg = aug(ecg)
    return ecg 