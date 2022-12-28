'''
CAP5404 Deep Learning for Computer Graphics
Project Part-1
Author: Pranath Reddy Kumbam
UFID: 8512-0977

- Train Classifiers on unbalanced Connect Four Data (MLP, Decision Tress, Random Forest)
'''

# Import libraries
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pickle

# Load Data
data = pd.read_csv('./Data/connectfour.data', sep= ',', header = None)
encoding = {"x":2, "o":1, "b":0, "win":0, "loss":1, "draw":2}
data = data.replace(encoding).to_numpy()
np.random.shuffle(data)

# Split the data into validation set
x, x_val, y, y_val = train_test_split(data[:,:-1], data[:,-1].reshape(-1), test_size=0.05)

# Shuffle the data
x, y = shuffle(x, y, random_state=0)
x_val, y_val = shuffle(x_val, y_val, random_state=0)

# Multi-Class Classification
print("Multi-Class Classification\n")
print("MLP\n")
MLP_Scores = []
fold_index = 1
for train_index, test_index in KFold(n_splits=10).split(x):
    print('Fold: ' + str(fold_index) + '\n')
    x_tr, x_ts = x[train_index], x[test_index]
    y_tr, y_ts = y[train_index], y[test_index]

    # MLP
    # Fitting and testing the model
    optimal_n = 300
    clf_MLP = MLPClassifier(hidden_layer_sizes=(optimal_n, optimal_n*2, optimal_n*2, optimal_n), max_iter=500)
    clf_MLP.fit(x_tr, y_tr)
    yp_MLP = clf_MLP.predict(x_ts)
    acc_MLP = accuracy_score(y_ts, yp_MLP)
    print("MLP Accuracy: " + str(acc_MLP) + '\n')
    MLP_Scores.append(acc_MLP)
    print("MLP Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts, yp_MLP, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")
    print("________________________________________________________  \n")

    fold_index += 1

print("Results: ")
print("MLP Accuracy: " + str(np.mean(MLP_Scores)) + " +/- " + str(np.std(MLP_Scores)))

print("Multi-Class Classification\n")
print("Decision Trees\n")
DT_Scores = []
fold_index = 1
for train_index, test_index in KFold(n_splits=10).split(x):
    print('Fold: ' + str(fold_index) + '\n')
    x_tr, x_ts = x[train_index], x[test_index]
    y_tr, y_ts = y[train_index], y[test_index]

    # Decision Tree
    # Fitting and testing the model
    clf_DT = DecisionTreeClassifier(min_samples_split=20)
    clf_DT.fit(x_tr, y_tr)
    yp_DT = clf_DT.predict(x_ts)
    acc_DT = clf_DT.score(x_ts, y_ts)
    print("DT Accuracy: " + str(acc_DT) + '\n')
    DT_Scores.append(acc_DT)
    print("DT Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts, yp_DT, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")
    print("________________________________________________________  \n")

    fold_index += 1

print("Results: ")
print("DT Accuracy: " + str(np.mean(DT_Scores)) + " +/- " + str(np.std(DT_Scores)))

print("Multi-Class Classification\n")
print("Random Forest\n")
RF_Scores = []
fold_index = 1
for train_index, test_index in KFold(n_splits=10).split(x):
    print('Fold: ' + str(fold_index) + '\n')
    x_tr, x_ts = x[train_index], x[test_index]
    y_tr, y_ts = y[train_index], y[test_index]

    # Random Forest
    # Fitting and testing the model
    # Used grid search
    clf_RF = RandomForestClassifier(max_depth=40, random_state=0)
    clf_RF.fit(x_tr, y_tr)
    yp_RF = clf_RF.predict(x_ts)
    acc_RF = clf_RF.score(x_ts, y_ts)
    print("RF Accuracy: " + str(acc_RF) + '\n')
    RF_Scores.append(acc_RF)
    print("RF Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts, yp_RF, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")
    print("________________________________________________________  \n")

    fold_index += 1

print("Results: ")
print("RF Accuracy: " + str(np.mean(RF_Scores)) + " +/- " + str(np.std(RF_Scores)))
