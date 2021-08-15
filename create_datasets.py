#Starting with BTC price data with 5 minute intervals between the observations, we want to
#create training sets with intervals of 10, 15, 20, 30, 60 minutes
import os
import numpy as np

btc_data = open('Datasets/Datasets_5/5_set.txt', 'r').readlines()
parent_dir = "/Users/bruker/PycharmProjects/Crypto_Price_Predictor/Datasets/"


#We have a set "data_k" with interval k between the observations, and want this to yield l = n/k
# sets with interval n
def NewDatasets(data_k, k, n):
    l = int(n/k)
    new_sets = []
    for i in range(l):
        new_sets.append([])

    for index, data in enumerate(data_k):
        for i in range(l):
            if index % l == i:
                new_sets[i].append(data)

    return new_sets

#We want to save these new datasets to files
def SaveDatasets(datasets, n):
    directory = "Datasets_"+str(n)
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    for index, set_ in enumerate(datasets):
        fileName = str(n)+"_Set_"+str(index)
        completeName = os.path.join(path, fileName)
        file1 = open(completeName, "x")
        np.savetxt(file1, set_, fmt="%s")
        file1.close()

#By combining the two functions above, we can now define a function to create and save new datasets
def CreateDatasets(data_k, k, n):
    new_datasets = NewDatasets(data_k, k, n)
    SaveDatasets(new_datasets, n)

CreateDatasets(btc_data, 5, 15)
CreateDatasets(btc_data, 5, 20)
CreateDatasets(btc_data, 5, 30)
CreateDatasets(btc_data, 5, 45)
CreateDatasets(btc_data, 5, 60)
