import random
import json
import argparse
import pickle
from collections import defaultdict
import torch
import torch.nn.functional as F


def stratified_indices(y, ratio, seed=0):
    rng = random.Random(seed)
    by_cl = defaultdict(list)
    for idx, label in enumerate(y):
        by_cl[label].append(idx)

    keep = []
    for cls, idxs in by_cl.items():
        k = max(1, int(len(idxs) * ratio))
        keep.extend(rng.sample(idxs, k))
    rng.shuffle(keep)
    return keep

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
        # Round up length to nearest multiple of 4
        pad_len = max_t
        if pad_len % 4 != 0:
            pad_len = ((pad_len + 3) // 4) * 4
    padded = []
    for m in stft_tensors:
        pad = pad_len - m.shape[1]
        padded.append(F.pad(m, (0, pad)))
    return torch.stack(padded), pad_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--split_data", type=str, default="../data/split_data.pkl")
    args = parser.parse_args()

    with open(args.split_data, "rb") as f:
        X_train_split, X_val_split, y_train_split, y_val_split = pickle.load(f)

    idxs = stratified_indices(y_train_split, args.ratio)
    json.dump(idxs, open(args.out, "w"))
    print(f"Saved {len(idxs)} indices → {args.out}")
