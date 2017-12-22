import numpy as np
from random import randint
import sys
import csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing



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

X_tr=[]
label=[]
for i in X_data:
    X_tr.append(i[:-1])
    label.append(i[-1])

#scaler = preprocessing.StandardScaler()
#X_tr   = scaler.fit_transform(X_tr)

#min_max_scaler = preprocessing.MinMaxScaler()
#X_tr= min_max_scaler.fit_transform(X_tr)

#max_abs_scaler = preprocessing.MaxAbsScaler()
#X_tr = max_abs_scaler.fit_transform(X_tr)


X_tr=X_tr[0:500]
label=label[0:500]

param_grid = [
  {'C': [1, 10, 100, 1000, 10000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000, 10000], 'gamma': [1e-2, 1e-3, 1e-4], 'kernel': ['rbf', 'poly']},
 ]

"""
param_grid = [
  {'estimator__C': [1, 10, 100, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000], 'estimator__gamma': [1e-2, 1e-3, 1e-4], 'estimator__kernel': ['rbf', 'poly', 'sigmoid', 'linear']},
 ]
"""

"""
param_grid = [
  {'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'estimator__gamma': [1e-3, 1e-2, 1e-1, 1], 'estimator__kernel': ['rbf', 'poly', 'sigmoid', 'linear']},
 ]
"""

"""
param_grid = [
  {'C': [1, 10, 80, 90, 100, 110, 120, 130, 1000, 10000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000, 9000, 10000, 11000], 'gamma': [1e-3, 1e-4], 'kernel': ['rbf', 'poly']},
 ]
"""

sample=[]
with open(sys.argv[2], 'rb') as f:
    reader=csv.reader(f.read().splitlines())
    for row in reader:
        sample.append(row)

for i in range(len(sample[0])):
    str_column_to_float(sample, i)

X_data=sample
X_val=[]
label1=[]
for i in X_data:
    X_val.append(i[:-1])
    label1.append(i[-1])

#X_val=X_val
#label1=label1
X_val=X_val[0:500]
label1=label1[0:500]
#X_val=min_max_scaler.transform(X_val)
#X_val=scaler.transform(X_val)
#X_val=max_abs_scaler.transform(X_val)

model_to_set=[SVC(decision_function_shape='ovo', kernel='rbf'), SVC(decision_function_shape='ovr', kernel='rbf')]
scores = ['precision', 'recall']


print("Tuning hyper-parameters")
for model in model_to_set:
    for score in scores:
        clf = GridSearchCV(model, param_grid, cv=3, n_jobs=6)
        clf.fit(X_tr, label)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = label1, clf.predict(X_val)
        print(classification_report(y_true, y_pred))
        print()

predict=clf.predict(X_tr)

count=0
for i in xrange(len(label)):
    if label[i]==predict[i]:
        count+=1

print " training accuracy:%f" % (float(count)/len(label))
    

predict=clf.predict(X_val)

count=0
for i in xrange(len(label1)):
    if label1[i]==predict[i]:
        count+=1

print " validation accuracy:%f" % (float(count)/len(label1))
