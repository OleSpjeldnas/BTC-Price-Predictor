import numpy as np
import torch
import torch.nn as nn
import sklearn
from sklearn.model_selection import train_test_split
import torch.utils.data
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt

#We begin by importing data and saving it to a list
directory = "Datasets/Datasets_5"
data_list = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        temp = open(f, "r").readlines()
        temp[:] = [x for x in temp if x != '\n']
        data_list += temp

data = np.asarray(data_list).astype(np.double)
#Find mean and std in order to scale data
mean = np.mean(data)
std = np.std(data)
data = (data - mean)/std

#Now implement sliding windows function in order to make training set

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


seq_length = 5

#Create input x of size 50 and corresponding y, make it into a tensor, and generate training and validation sets
x, y = sliding_windows(data, seq_length)
x = torch.tensor(np.array(x), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
last_measurements = []
l = x_test.data.numpy()
for i, el in enumerate(l):
    e = el[-1]
    last_measurements.append(e)

x_train = Variable(x_train.view(-1, 1, seq_length))
x_test = Variable(x_test.view(-1, 1, seq_length))
y_train = Variable(y_train.view(-1, 1))
y_test = Variable(y_test.view(-1, 1))


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


num_epochs = 200

input_size = seq_length
hidden_size = 20
num_layers = 1
output_size = 1

price_predictor = LSTM(output_size, input_size, hidden_size, num_layers)
y_test = y_test.data.numpy()
test_up = list()
#Train the NN
def fit(lstm, input, y_true,  num_epochs):
    criterion = torch.nn.MSELoss()
    train_hist = list()
    #test_hist = list()
    learning_rate = 1e-6
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = lstm(input)
        optimizer.zero_grad()

        loss = criterion(outputs, y_true)
        train_hist.append(loss.item())
        test_hist = price_predictor(x_test).data.numpy()
        guessed_up = 0
        true_price_up = 0

        for index, guess in enumerate(test_hist):
            # We look at the price development from the last measurement taken before prediction
            last_measurement = last_measurements[index]
            # The predicted change is how much positive or negative change our model predicts
            predicted_change = guess - last_measurement
            # The real change is the ground truth - how much did the price change
            real_change = y_test[index] - last_measurement

            if predicted_change > 0:
                guessed_up += 1
                if real_change > 0:
                    true_price_up += 1
        test_up.append(true_price_up/guessed_up)
        loss.backward()

        optimizer.step()
        if epoch % 10 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    return train_hist


#Train network and feed it the test set
hist = fit(price_predictor, x_train, y_train, num_epochs)
test_hist = price_predictor(x_test).data.numpy()
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
    last_measurement = last_measurements[index]
    #The predicted change is how much positive or negative change our model predicts
    predicted_change = guess - last_measurement
    #The real change is the ground truth - how much did the price change
    real_change = y_test[index] - last_measurement


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
                 "Relative Correct Guesses: ": trues/len(y_test),
                 "Guessed Up, Went Up: ": true_price_up,
                 "Guessed Down, Went Down: ": true_price_down,
                 "Guessed Up, Went Down: ": assume_up_price_down,
                 "Guessed Down, Went Up: ": assume_down_price_up
                 }


print(final_verdict)

PATH = "Models/5_price_predictor_15_hidden_20.pth"
torch.save(price_predictor.state_dict(), PATH)

