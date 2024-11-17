import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import glob
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, time_series, time_seriesVol, sequence_length):
        self.time_series = time_series
        self.time_seriesVol = time_seriesVol
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.time_series) - self.sequence_length

    def __getitem__(self, idx):
        price_sequence = torch.tensor(self.time_series[idx : idx + self.sequence_length],dtype=torch.float32)
        volume_sequence = torch.tensor(self.time_seriesVol[idx : idx + self.sequence_length], dtype = torch.float32)
        return price_sequence

def modelRNN(price_history_len = 9):
    model = nn.Linear(price_history_len, 1, bias = True)
    return model

def loss(pred, y):
    """Mean Absolute Error loss"""
    return (pred - y).abs().mean()

def train(model, loss, dataloader, optimizer):
    """Helper function to train our model."""
    total_error = 0.
    for it, sequences in enumerate(dataloader):
        optimizer.zero_grad()
        # Prepare model inputs and targets
        price_history = sequences[: ,:-1]
        target = sequences[:,-1]

        # Compute model predictions
        pred = model(price_history)
        pred = pred.flatten()

        # Compute the loss
        l = loss(pred, target)
        total_error += l.item()

        # Update the weights
        l.backward()
        optimizer.step()

    return total_error / len(dataloader)

def fit(model, loss, dataloader, epochs=300, lr = 0.001):
    prev_err = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(epochs):
        err = train(model, loss, dataloader, optimizer)
        print(f"[Ep{ep}] Error {err:.3f}")
        if abs(err - prev_err) < 0.01 and err < 0.236:
            break
        prev_err = err

def startTrain(model, price, volume, date):
    #Hyperparameters
    price_history_len = 9
    if len(price) < 200:
        batch_size = 2
    elif len(price) < 400:
        batch_size = 4
    elif len(price) < 600:
        batch_size = 8
    else:
        batch_size = 16
    epochs = 300

    dataset = TimeSeriesDataset(price, volume, price_history_len+1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    fit(model, loss, dataloader, epochs=epochs)
    graphIt(dataset, model, price, date)

    predict_and_graph(model, dataset)

def create_model(filename):
    df = pd.read_csv(filename)
    time_series = df['Close'].values
    time_seriesVol = df['Volume'].values
    date = df['Date'].values

    model = modelRNN()
    name = os.path.basename(filename)

    startTrain(model, time_series, time_seriesVol,date)

    torch.save(model.state_dict(), os.path.join('models', f"{os.path.basename(name[:name.rfind('.')])}.pth"))
    return model

def main():
    dir_path = 'models'

    pth_files = glob.glob(dir_path + "/*.pth")
    for pth_file in pth_files:
        os.remove(pth_file)
        print(f"Deleted {pth_file}")

    csv_list = glob.glob('datasets_chosen/*.csv')

    for csv_file in csv_list[:3]:
        model = create_model(csv_file)
        

def graphIt(dataset, model, price, date):
    with torch.no_grad():
        predictions, errors = [], []
        for i in range(len(dataset)):
            sequence = dataset[i]
            past, price_gt = sequence[:-1], sequence[-1]
            price_pred = model(past)

            err = price_pred - price_gt

            errors.append(err.item())
            predictions.append(price_pred.item())

    plt.plot([None]*9+predictions, label='prediction')
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

def predict_and_graph(model, dataset, initial_sequence_len=9, num_predictions=100):
    with torch.no_grad():
        # Initial sequence for prediction
        initial_sequence = dataset[0][:initial_sequence_len]  # Take the first sequence from the dataset

        # Prepare the data for prediction
        predictions = []
        sequence = initial_sequence.tolist()  # Convert tensor to list for further manipulation

        # Predict the next 100 points
        for _ in range(num_predictions):
            # Prepare the current sequence (last `initial_sequence_len` points)
            current_input = torch.tensor(sequence[-initial_sequence_len:], dtype=torch.float32).unsqueeze(0)
            
            # Get the model's prediction for the next point
            next_point = model(current_input).item()
            predictions.append(next_point)
            
            # Add the predicted point to the sequence for future predictions
            sequence.append(next_point)
        
        # Graph the initial sequence along with the predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(range(initial_sequence_len), dataset[0][:initial_sequence_len].numpy(), label='Initial Sequence', color='blue')
        plt.plot(range(initial_sequence_len, initial_sequence_len + num_predictions), predictions, label='Predicted Sequence', color='red')
        plt.ylabel('Price')
        plt.xlabel('Time Step')
        plt.title(f"Initial Sequence and Predicted Values for {num_predictions} Steps")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()