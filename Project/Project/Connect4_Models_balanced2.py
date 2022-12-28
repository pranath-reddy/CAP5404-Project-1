'''
CAP5404 Deep Learning for Computer Graphics
Project Part-1
Author: Pranath Reddy Kumbam
UFID: 8512-0977

- Train Classifiers on balanced Connect Four Data (KNN, SVM)
'''

# Import libraries
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import pickle

# Load Data
data = pd.read_csv('./Data/connectfour.data', sep= ',', header = None)
encoding = {"x":2, "o":1, "b":0, "win":0, "loss":1, "draw":2}
data = data.replace(encoding).to_numpy()

# Balance the Data based on Class with lowest No Of samples
data_win = np.asarray([sample for sample in data if sample[-1] == 1])
data_loss = np.asarray([sample for sample in data if sample[-1] == 0])
data_draw = np.asarray([sample for sample in data if sample[-1] == 2])
data = (np.concatenate((data_win[:data_draw.shape[0]], data_loss[:data_draw.shape[0]], data_draw)))
np.random.shuffle(data)

# Split the data into validation set
x, x_val, y, y_val = train_test_split(data[:,:-1], data[:,-1].reshape(-1), test_size=0.05)

# Shuffle the data
x, y = shuffle(x, y, random_state=0)
x_val, y_val = shuffle(x_val, y_val, random_state=0)

# Multi-Class Classification
print("Multi-Class Classification\n")
print("KNN\n")
KN_Scores = []
fold_index = 1
for train_index, test_index in KFold(n_splits=10).split(x):
    print('Fold: ' + str(fold_index) + '\n')
    x_tr, x_ts = x[train_index], x[test_index]
    y_tr, y_ts = y[train_index], y[test_index]

    # KNN
    # Fitting and testing the model
    clf_KN = KNeighborsClassifier(n_neighbors=1)
    clf_KN.fit(x_tr, y_tr)
    yp_KN = clf_KN.predict(x_ts)
    acc_KN = accuracy_score(y_ts, yp_KN)
    print("KN Accuracy: " + str(acc_KN) + '\n')
    KN_Scores.append(acc_KN)
    print("KN Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts, yp_KN, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")
    print("________________________________________________________  \n")

    fold_index += 1

print("Results: ")
print("KN Accuracy: " + str(np.mean(KN_Scores)) + " +/- " + str(np.std(KN_Scores)))

print("Multi-Class Classification\n")
print("Decision Trees\n")
SV_Scores = []
fold_index = 1
for train_index, test_index in KFold(n_splits=10).split(x):
    print('Fold: ' + str(fold_index) + '\n')
    x_tr, x_ts = x[train_index], x[test_index]
    y_tr, y_ts = y[train_index], y[test_index]

    # SVM
    # Fitting and testing the model
    clf_SV = LinearSVC(tol=1e-5)
    clf_SV.fit(x_tr, y_tr)
    yp_SV = clf_SV.predict(x_ts)
    acc_SV = clf_SV.score(x_ts, y_ts)
    print("SV Accuracy: " + str(acc_SV) + '\n')
    SV_Scores.append(acc_SV)
    print("SV Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts, yp_SV, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")
    print("________________________________________________________  \n")

    fold_index += 1

print("Results: ")
print("SV Accuracy: " + str(np.mean(SV_Scores)) + " +/- " + str(np.std(SV_Scores)))
