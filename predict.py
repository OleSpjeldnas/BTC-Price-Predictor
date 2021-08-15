import torch
import torch.nn as nn
import numpy as np
import os
from coinpaprika import client as Coinpaprika
import pandas as pd
import time
from torch.autograd import Variable
#Import the most recent 24 Hour Data
filename = "CoinPaprika_BTC_price_24h_2021-08-13.csv"
training_data = pd.read_csv(filename)
price_array = []

for i, row in training_data.iterrows():
    price_array.append(row['Price'])

#os.remove(filename)
client = Coinpaprika.Client()


std = np.std(price_array)
mean = np.mean(price_array)
price_array = (price_array-mean)/std


class LSTM(nn.Module):

    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=0)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)
        return out


seq_length = 50
input_size = seq_length
hidden_size = 80
num_layers = 1
output_size = 1
#Load model
predictor_5 = LSTM(output_size, input_size, hidden_size, num_layers)
predictor_5.load_state_dict(torch.load("Models/5_price_predictor.pth"))

input = price_array[-50:]
input = Variable(torch.tensor(input, dtype=torch.float32).view(-1, 1, 50))
output = predictor_5(input).data.numpy()
btc_old = price_array[-1]
pred_change = ((output - btc_old) / btc_old) * 100
pred = [[str(pred_change) + " %"]]
df = pd.DataFrame(pred)
df.to_csv("Data For Predictions/Predictions.csv")

while True:
    btc = (client.ticker("btc-bitcoin")["quotes"]["USD"]["price"]-mean)/std

    if btc == btc_old:
        time.sleep(10)
    else:
        actual_change = ((btc - btc_old) / btc_old) * 100
        np.append(price_array, btc)

        pred_actual = [[str(pred_change) + " %", str(actual_change) + " %"]]
        df = pd.DataFrame(pred_actual)
        df.to_csv("Data For Predictions/Predictions_vs_Reality.csv")

        input = price_array[-50:]
        input = Variable(torch.tensor(input, dtype=torch.float32).view(1, -1, 50))
        output = predictor_5(input).data.numpy()
        pred_change = ((output - btc_old) / btc_old) * 100
        btc_old = btc

        pred = [[str(pred_change) + " %"]]
        df = pd.DataFrame(pred)
        df.to_csv("Data For Predictions/Predictions.csv")
        time.sleep(300)


