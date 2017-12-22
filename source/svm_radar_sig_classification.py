import numpy as np
from random import randint
import sys
import csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

def confusion_matrix_t(y_true, y_pred):
    print "confusion matrix:"
    print(confusion_matrix(y_true, y_pred))



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

print label.count(0)
print label.count(1)
scaler = preprocessing.StandardScaler()
X_tr   = scaler.fit_transform(X_tr)

#min_max_scaler = preprocessing.MinMaxScaler()
#X_tr= min_max_scaler.fit_transform(X_tr)

#max_abs_scaler = preprocessing.MaxAbsScaler()
#X_tr = max_abs_scaler.fit_transform(X_tr)

param_grid = [
  {'C': [1, 10, 100, 1000, 10000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000, 10000], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6], 'kernel': ['rbf']},
 ]

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

print label1.count(0)
print label1.count(1)


X_val=scaler.transform(X_val)
#X_val=min_max_scaler.transform(X_val)
#X_val=max_abs_scaler.transform(X_val)

scores = ['precision', 'recall']
print("Tuning hyper-parameters")
for score in scores:
    clf = GridSearchCV(SVC(), param_grid, cv=5, scoring='%s_macro' % score)
    #clf = SVC(C=1.0, kernel="linear")
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

print "training accuracy:%f" % (float(count)/len(label))
print ""
print "Confusion Matrix:Training Set"
confusion_matrix_t(label, predict)


predict=clf.predict(X_val)

count=0
for i in xrange(len(label1)):
    if label1[i]==predict[i]:
        count+=1

print "validation accuracy:%f" % (float(count)/len(label1))

print ""
print "Confusion Matrix: Validation Set"
confusion_matrix_t(label1, predict)







