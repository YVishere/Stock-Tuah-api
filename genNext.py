import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import shutil
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesDataset(Dataset):
    def __init__(self, time_series, time_seriesVol, sequence_length):
        self.time_series = time_series
        self.time_seriesVol = time_seriesVol
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.time_series) - self.sequence_length

    def __getitem__(self, idx):
        price_sequence = torch.tensor(self.time_series[idx : idx + self.sequence_length], dtype=torch.float32).unsqueeze(-1)
        volume_sequence = torch.tensor(self.time_seriesVol[idx : idx + self.sequence_length], dtype=torch.float32).unsqueeze(-1)
        sequence = torch.cat([price_sequence, volume_sequence], dim=-1)
        return sequence

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, output_size=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_lstm_out = lstm_out[:, -1, :]
        output = self.fc(last_lstm_out)
        return output
    
def gen_next(base, x, modelName):
    price_scaler = MinMaxScaler(feature_range=(0,1))
    volume_scaler = MinMaxScaler(feature_range=(0,1))
    price= price_scaler.fit_transform(base['Close'].values.reshape(-1,1))
    volume = volume_scaler.fit_transform(base['Volume'].values.reshape(-1,1))
    dataset = TimeSeriesDataset(price, volume, 10)
    model_path = f'models/{modelName}.pth'

    model = LSTMModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    toUse = dataset[len(dataset)-1]
    toUse = toUse[1:]
    toUse = toUse.squeeze(1)
    alpha = 0.8
    future = []
    futureVol = []
    print(toUse.shape)
    print(toUse)
    for i in range(x):
        toUse = toUse.unsqueeze(0)
        pred = model(toUse)
        price_pred = pred[-1,0].item()
        price_pred = alpha * price_pred + (1 - alpha) * toUse[-1, 0, 0].item()
        volume_pred = pred[-1,1].item()
        future.append(price_pred)
        futureVol.append(volume_pred)
        toUse = toUse.squeeze(0)
        toUse = toUse[1:]
        print(toUse.shape)
        toUse = torch.cat([toUse, torch.tensor([[price_pred, volume_pred]], dtype=torch.float32)], dim=0)

    future = price_scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()
    futureVol = volume_scaler.inverse_transform(np.array(futureVol).reshape(-1, 1)).flatten()

    return future, futureVol

def copy_datasets(src_dir, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Get a list of all CSV files in the source directory
    csv_files = glob.glob(os.path.join(src_dir, "*.csv"))

    # Copy each file to the destination directory
    for csv_file in csv_files:
        shutil.copy(csv_file, dest_dir)
        print(f"Copied {csv_file} to {dest_dir}")

def append_to_csv(file_path, future, futureVol):
    # Create a DataFrame from the predictions
    df = pd.DataFrame({
        'Date': "future",
        'Close': future,
        'Volume': futureVol
    })
    df.to_csv(file_path, mode='a', header=False, index=False)
    
def main(x):
    dir_path = 'modded_datasets'

    base = None

    csv_files = glob.glob(dir_path + "/*.csv")
    
    if len(csv_files) == 0:
        copy_datasets('datasets_chosen', 'modded_datasets')
        csv_files = glob.glob(dir_path + "/*.csv")

    for csv_file in csv_files:
        base = pd.read_csv(csv_file)
        p, v = gen_next(base, x, os.path.basename(csv_file)[:os.path.basename(csv_file).rfind('.')])
        append_to_csv(csv_file, p, v)
        print(f"Modified {csv_file}")
    
    return "Done"
