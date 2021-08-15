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

#Use the following two lines in order to find out the hidden size of a model
#state_dict = torch.load("Models/5_price_predictor_50_hidden_50.pth")
#print(state_dict['fc.weight'].shape)


#Load all models
predictor_5 = LSTM(output_size=1, input_size=5, hidden_size=150, num_layers=1)
predictor_5.load_state_dict(torch.load("Models/5_price_predictor_5_hidden_150.pth"))
predictor_10 = LSTM(output_size=1, input_size=10, hidden_size=20, num_layers=1)
predictor_10.load_state_dict(torch.load("Models/5_price_predictor_10_hidden_20.pth"))
predictor_15 = LSTM(output_size=1, input_size=15, hidden_size=20, num_layers=1)
predictor_15.load_state_dict(torch.load("Models/5_price_predictor_15_hidden_20.pth"))
predictor_20 = LSTM(output_size=1, input_size=20, hidden_size=50, num_layers=1)
predictor_20.load_state_dict(torch.load("Models/5_price_predictor_20_hidden_50.pth"))
predictor_50 = LSTM(output_size=1, input_size=50, hidden_size=50, num_layers=1)
predictor_50.load_state_dict(torch.load("Models/5_price_predictor_50_hidden_50.pth"))

#Create evaluation sets on which to test the data
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def create_sets(seq_l):
    x, y = sliding_windows(price_array, seq_l)
    new_y = list()
    last_measurements = list()
    for index, element in enumerate(y):
        last_measurements.append(x[index][-1])
        if element < x[index][-1]:
            new_y.append(0)
        else:
            new_y.append(1)
    x = torch.tensor(np.array(x), dtype=torch.float32)
    new_y = torch.tensor(np.array(new_y), dtype=torch.float32)
    return Variable(x.view(-1, 1, seq_l)), y, last_measurements

x_5, y_5, last_measurements_5 = create_sets(5)
x_10, y_10, last_measurements_10 = create_sets(10)
x_15, y_15, last_measurements_15 = create_sets(15)
x_20, y_20, last_measurements_20 = create_sets(20)
x_50, y_50, last_measurements_50 = create_sets(50)

y_eval_5 = predictor_5(x_5).data.numpy()
y_eval_10 = predictor_10(x_10).data.numpy()
y_eval_15 = predictor_15(x_15).data.numpy()
y_eval_20 = predictor_20(x_20).data.numpy()
y_eval_50 = predictor_50(x_50).data.numpy()

#print(y_5[5:10])
#print(y_10[:5])


#Change our predictions into binary up/down predictions
def up_or_down(last_measurements, y):
    binary = list()
    for index, element in enumerate(last_measurements):
        if element < y[index]:
            binary.append(0)
        else:
            binary.append(1)
    return binary

#y_binary_5 = up_or_down(last_measurements_5, y_eval_5)
#y_binary_10 = up_or_down(last_measurements_10, y_eval_10)
#y_binary_15 = up_or_down(last_measurements_15, y_eval_15)
#y_binary_20 = up_or_down(last_measurements_20, y_eval_20)
#y_binary_50 = up_or_down(last_measurements_50, y_eval_50)

#Evaluate the collective performance of models
def evaluate_models(y_1, y_2, y):
    guess_true = 0
    guess_true_and_true = 0
    l = len(y_2)-len(y_1)
    if l < 0:
        y_1[:] = y_1[-l:]
    else:
        y_2[:] = y_2[l:]
    for index, guess in enumerate(y_1):
        if guess == 1 and y_2[index] == 1:
            guess_true += 1
            if y[index] == 1:
                guess_true_and_true += 1

    return guess_true_and_true/guess_true, guess_true

#ratio, total_guesses = evaluate_models(y_binary_20, y_binary_20, y_20)
#print("Ratio: "+str(ratio*100) + " %")
#print("Total Guesses: " + str(total_guesses))


test_hist = predictor_50(x_50).data.numpy()
trues = 0
total_trues = 0
guessed_up = 0
falses = 0
true_price_up = 0
true_price_down = 0
assume_up_price_down = 0
assume_down_price_up = 0

for index, guess in enumerate(test_hist):
    #We look at the price development from the last measurement taken before prediction
    last_measurement = last_measurements_50[index]
    #The predicted change is how much positive or negative change our model predicts
    predicted_change = guess - last_measurement
    #The real change is the ground truth - how much did the price change
    real_change = y_50[index] - last_measurement


    if predicted_change > 0:
        guessed_up += 1
        if real_change > 0:
            trues += 1
            true_price_up += 1
            total_trues +=1
        elif real_change <= 0:
            falses += 1
            assume_up_price_down += 1

    elif predicted_change < 0:
        if real_change < 0:
            trues += 1
            true_price_down += 1
        elif real_change >= 0:
            falses += 1
            assume_down_price_up += 1
            total_trues +=1

final_verdict = {"True Guesses: ": trues,
                 "Real Trues: ": true_price_up/(true_price_up+assume_up_price_down),
                 "False Guesses: ": falses,
                 "Relative Correct Guesses: ": trues/len(y_50),
                 "Guessed Up, Went Up: ": true_price_up,
                 "Guessed Down, Went Down: ": true_price_down,
                 "Guessed Up, Went Down: ": assume_up_price_down,
                 "Guessed Down, Went Up: ": assume_down_price_up
                 }


print(final_verdict)