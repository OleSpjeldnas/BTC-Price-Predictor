import pandas as pd
import numpy as np
import os

#Takes in csv file and converts it to a txt file with corresponding price list


training_data = pd.read_csv('')
val_array = []


for i, row in training_data.iterrows():
    val_array.append(row['Price'])

#a_file = open("5_set.txt", "w")
#np.savetxt(a_file, val_array, fmt="%s")

#a_file.close()
