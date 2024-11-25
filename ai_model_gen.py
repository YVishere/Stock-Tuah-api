import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import glob
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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
    def __init__(self, input_size=2, hidden_size=256, num_layers=2, output_size=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_lstm_out = lstm_out[:, -1, :]
        output = self.fc(last_lstm_out)
        return output

def train(model, loss, dataloader, optimizer):
    total_error = 0.
    for it, sequences in enumerate(dataloader):
        optimizer.zero_grad()
        price_history = sequences[:, :-1]
        target = sequences[:, -1].flatten()

        pred = model(price_history)
        pred = pred.flatten()

        l = loss(pred, target)
        total_error += l.item()

        l.backward()
        optimizer.step()

    return total_error / len(dataloader)

def fit(model, loss, dataloader, epochs=300, lr=0.0001):
    prev_err = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(epochs):
        err = train(model, loss, dataloader, optimizer)
        print(f"[Ep{ep}] Error {err}")

def startTrain(model, price, volume, date, price_scaler):
    price_history_len = 9
    if len(price) < 200:
        batch_size = 2
    elif len(price) < 400:
        batch_size = 4
    elif len(price) < 600:
        batch_size = 8
    else:
        batch_size = 16
    epochs = 8

    dataset = TimeSeriesDataset(price, volume, price_history_len + 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loss = nn.MSELoss()
    fit(model, loss, dataloader, epochs=epochs)
    
    graphIt(dataset, model, price, date, price_scaler)

def create_model(filename):
    df = pd.read_csv(filename)
    time_series = df['Close'].values
    time_seriesVol = df['Volume'].values
    date = df['Date'].values

    price_scaler = MinMaxScaler(feature_range=(0,1))
    time_series = price_scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
    volume_scaler = MinMaxScaler(feature_range=(0,1))
    time_seriesVol = volume_scaler.fit_transform(time_seriesVol.reshape(-1, 1)).flatten()

    model = LSTMModel()

    name = os.path.basename(filename)

    startTrain(model, time_series, time_seriesVol, date, price_scaler)

    torch.save(model.state_dict(), os.path.join('models', f"{os.path.basename(name[:name.rfind('.')])}.pth"))
    return model

def graphIt(dataset, model, price, date, price_scaler):
    with torch.no_grad():
        predictions, errors = [], []
        for i in range(len(dataset)):
            sequence = dataset[i]
            past, price_gt = sequence[:-1], sequence[-1, 0]
            past = past.unsqueeze(0)
            price_pred = model(past)
            price_pred = price_pred[-1,0].item()
            err = price_pred - price_gt
            errors.append(err)
            predictions.append(price_pred)

    predictions = price_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    price = price_scaler.inverse_transform(np.array(price).reshape(-1, 1)).flatten()

    future = []
    toUse = dataset[0]
    toUse = toUse[1:]
    alpha = 0.8
    for i in range(6000):
        toUse = toUse.unsqueeze(0)
        pred = model(toUse)
        price_pred = pred[-1,0].item()
        price_pred = alpha * price_pred + (1 - alpha) * toUse[-1, 0, 0].item()
        volume_pred = pred[-1,1].item()
        future.append(price_pred)
        toUse = toUse.squeeze(0)
        toUse = toUse[1:]
        toUse = torch.cat([toUse, torch.tensor([[price_pred, volume_pred]], dtype=torch.float32)], dim=0)

    future = price_scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()

    plt.plot([None]*9+predictions.tolist(), label='prediction')
    plt.plot(price, label='ground truth')
    plt.ylabel('Bitcoin Price [$]')
    plt.gca().set_xticklabels(date, rotation=30)
    plt.legend()
    plt.show()

    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel('Error [$]')
    plt.ylabel('Frequency')
    plt.title('Histogram of Errors')
    plt.show()

    plt.plot(future, label='future')
    plt.show()

def main():
    dir_path = 'models'

    pth_files = glob.glob(dir_path + "/*.pth")
    for pth_file in pth_files:
        os.remove(pth_file)
        print(f"Deleted {pth_file}")

    csv_list = glob.glob('datasets_chosen/*.csv')

    for ind,csv_file in enumerate(csv_list[:1]):
        print(f"Training model {ind+1}/{len(csv_list)}")
        model = create_model(csv_file)

if __name__ == '__main__':
    main()