import numpy as np
from random import randint
import sys
import csv



def str_column_to_float(X_data, column):
    for row in X_data:
        row[column]=float(row[column].strip())

sample=[]
with open(sys.argv[1], 'rb') as f:
    reader=csv.reader(f.read().splitlines())
    for row in reader:
        sample.append(row)

for i in range(len(sample[0])):
    str_column_to_float(sample, i)

X_data=sample


np.random.shuffle(X_data)

    
with open('train.csv', 'w') as f:
    wr=csv.writer(f)
    wr.writerows(X_data[0:280])

with open('validation.csv', 'w') as f:
    wr=csv.writer(f)
    wr.writerows(X_data[281:])

"""
with open('test.csv', 'w') as f:
    wr=csv.writer(f)
    wr.writerows(X_data[317:])

"""

