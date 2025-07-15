import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
from .model import ECGCNN
from .data import pad_sequences
from .utils import stft_transform
from .ecg_parser import read_zip_binary

def run_inference(
    model_path,
    test_zip_path,
    output_path="base.csv",
    batch_size=32,
    device=None,
    pad_len=18286
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class ECGTestDataset(Dataset):
        def __init__(self, signals):
            self.lengths = [len(x) for x in signals]
            # Padding auf Trainings-Länge
            self.padded = pad_sequences(signals, max_len=pad_len)

        def __len__(self):
            return len(self.padded)

        def __getitem__(self, idx):
            return torch.tensor(self.padded[idx], dtype=torch.float32), self.lengths[idx]

    print("Load testdata..")
    raw_test_signals = read_zip_binary(test_zip_path)
    test_dataset = ECGTestDataset(raw_test_signals)
 # Prüfe die Länge der gepaddeten Sequenzen
    padded = test_dataset.padded
    print("Length of padded sequences:", set(len(seq) for seq in padded))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Load model...")
    model = ECGCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Start inference...")
    all_preds = []

    # Finde die erwartete Zeitlänge (aus der ersten Batch)
    expected_time_dim = None

    with torch.no_grad():
        for X_batch, lengths in test_loader:
            X_batch = X_batch.to(device)
            lengths = torch.tensor(lengths).to(device)
            X_stft = stft_transform(X_batch, lengths, n_fft=256, hop_length=128)
            # print("X_stft shape:", X_stft.shape)
            # Padding der letzten Batch auf gleiche Zeitlänge
            if expected_time_dim is None:
                expected_time_dim = X_stft.shape[-1]
            if X_stft.shape[-1] < expected_time_dim:
                pad_width = expected_time_dim - X_stft.shape[-1]
                X_stft = F.pad(X_stft, (0, pad_width))
            outputs = model(X_stft)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    print("Save base.csv...")
    df = pd.DataFrame({"y": all_preds})
    df.to_csv(output_path, index=False)
    print(f"Finished. Results are sved in {output_path}")