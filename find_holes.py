import pandas as pd

#This script analyses the csv data to make sure that there are no holes in it, e.g. that each consecutive value is taken
# 5 minutes after the previous one

interval = 300000
training_data = pd.read_csv('5_min_30.07,14.30-....csv')
prev = 0


for i, row in training_data.iterrows():
    if i > 0:
        if row['Category'] - prev != interval:
            print(i)
            break
    prev = row['Category']