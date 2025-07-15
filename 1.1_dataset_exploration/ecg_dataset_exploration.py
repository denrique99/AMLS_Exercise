import struct
import zipfile
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

def read_zip_binary(path: str) -> List[List[int]]:
    """Reads a binary zip file containing a ragged array of shorts."""
    ragged_array = []
    with zipfile.ZipFile(path, 'r') as zf:
        inner_path = path.split("/")[-1].split(".")[0]
        with zf.open(f'{inner_path}.bin', 'r') as r:
            read_binary_from(ragged_array, r)
    return ragged_array

def read_binary_from(ragged_array, r):
    while True:
        size_bytes = r.read(4)
        if not size_bytes:
            break
        sub_array_size = struct.unpack('i', size_bytes)[0]
        sub_array = list(struct.unpack(f'{sub_array_size}h', r.read(sub_array_size * 2)))
        ragged_array.append(sub_array)

def analyze_ecg_data(ecg_data: List[List[int]], labels: List[int]) -> pd.DataFrame:
    lengths = [len(x) for x in ecg_data]
    df_stats = pd.DataFrame({
        "length": lengths,
        "label": labels
    })
    stats_summary = df_stats.groupby("label")["length"].count().reset_index(name="count")

    sampling_rate = 300  # Hz
    start_sample = 3000
    end_sample = 5000

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle("Zoomed ECG Signal (samples 3000–5000) per Class")
    for i in range(4):
        idx = df_stats[df_stats["label"] == i].index[0]
        example = ecg_data[idx]
        time_axis = np.arange(start_sample, end_sample) / sampling_rate
        axs[i // 2][i % 2].plot(time_axis, example[start_sample:end_sample])
        axs[i // 2][i % 2].set_title(f"class {i}")
        axs[i // 2][i % 2].set_xlabel("Time (s)")
        axs[i // 2][i % 2].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    return stats_summary

def split_and_save_data(train_data: List[List[int]], labels: List[int], output_path: str):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        train_data,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    with open(output_path, "wb") as f:
        pickle.dump((X_train_split, X_val_split, y_train_split, y_val_split), f)

    print(f"Training-data: {len(X_train_split)} time series")
    print(f"Validation-data: {len(X_val_split)} time series")
    print("Train-class-distribution:", Counter(y_train_split))
    print("Val-class-distribution:", Counter(y_val_split))

def main():
    # Labels einlesen
    labels = pd.read_csv("data/y_train.csv", header=None)[0].tolist()

    # Daten einlesen
    train_data = read_zip_binary("data/X_train.zip")

    # Prüfung
    if isinstance(train_data, list) and len(train_data) > 0 and isinstance(train_data[0], list):
        print(f"Successfully loaded: {len(train_data)} time series")
        print(f"→ First time series: {len(train_data[0])} values, example: {train_data[0][:10]}")
    else:
        print("Error loading binary file.")
        return

    # Analysis & Visualization
    summary = analyze_ecg_data(train_data, labels)
    print("Class-Distribution:\n", summary)

    # Split & Save
    split_and_save_data(train_data, labels, "1.2_and_3_tuning_and_augmentation/data/split_data.pkl")

if __name__ == "__main__":
    main()
