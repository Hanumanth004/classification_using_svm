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

#print X_data


np.random.shuffle(X_data)

"""
temp=X_data[0:280]
label1=[]
for i in temp:
    label1.append(i[-1])
print label1.count(0)
print label1.count(1)
"""
    
with open('train_vowel.csv', 'w') as f:
    wr=csv.writer(f)
    wr.writerows(X_data[0:791])

with open('validation_vowel.csv', 'w') as f:
    wr=csv.writer(f)
    wr.writerows(X_data[792:])
"""
with open('test_vowel.csv', 'w') as f:
    wr=csv.writer(f)
    wr.writerows(X_data[892:])
"""

"""
sample=[]
with open(sys.argv[1], 'rb') as f:
    reader=csv.reader(f.read().splitlines())
    for row in reader:
        sample.append(row)


X_tr=sample
X_tr=np.array(X_tr)
X_tr=X_tr.astype(np.float)


sample=[]
with open(sys.argv[2], 'rb') as f:
    reader=csv.reader(f.read().splitlines())
    for row in reader:
        sample.append(row)

X_val=sample
X_val=np.array(X_val)
X_val=X_val.astype(np.float)


label1=[]
for i in X_tr:
    label1.append(i[-1])
print label1.count(0)
print label1.count(1)
"""
