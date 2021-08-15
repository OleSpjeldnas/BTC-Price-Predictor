# BTC-Price-Predictor
A neural network-based model for predicting future prices of Bitcoin. The neural networks are LSTMs, whereby the model takes in an input of size n consisting of previous price points with an equal time difference between each of them. I chose to use 5-minute data as a basis.
I used data from a 10 day period to train and validate the models on, varying the input size from 5 to 50. By tweaking the hyperparameters I got promising results, with the resulting models having 52-55% accuracy in predicting whether price would go up or down. 
However, 

"gather_data.py" uses the Coinpaprika API in order to get real-time BTC price data and saving it to a .txt file.

"find_holes.py" takes in a Coinpaprika csv file of past price data and notifies you of if and where there are holes in the data, e.g. if more than 5 minutes have passed between the i'th and the (i+1)'st entry.

"csv_to_txt.py" takes in a Coinpaprika csv and puts out a .txt file with the corresponding price data. This is somewhat superfluous, but I had already written the "gather_data.py" script, and this was the easiest way to combine data in the different formats.

"train_models.py" uses a sliding windows function to create training sets using the data provided, then defines an LSTM neural network architecture and trains the model on the training sets. Finally, the model is evaluated on the validation set.

"test_models.py" imports the given model(s) and creates corresponding testing sets based on input size, before checking the accuracy of the model(s) on these sets.

"predict.py" is now where the finished model is imported and being fed real-time price data in order to continuously make predictions.
