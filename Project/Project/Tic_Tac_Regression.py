'''
CAP5404 Deep Learning for Computer Graphics
Project Part-1
Author: Pranath Reddy Kumbam
UFID: 8512-0977

- Train regression models on the Tic Tac Toe multi-label dataset (MLP, KNN)
'''

# Import libraries
from sklearn.utils import shuffle
import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import multilabel_confusion_matrix
import pickle

# Load Data
data_multi = np.loadtxt('./Data/tictac_multi.txt')[:, :9]
labels_multi = np.loadtxt('./Data/tictac_multi.txt')[:, 9:]

# Split the data for 1/10th of the dataset
x_multi, x_val_multi, y_multi, y_val_multi = train_test_split(data_multi, labels_multi, test_size=0.05)

# Shuffle the data
x_multi, y_multi = shuffle(x_multi, y_multi, random_state=0)
x_val_multi, y_val_multi = shuffle(x_val_multi, y_val_multi, random_state=0)

# MLP for Multi-Label
# Architecture from Classification
print("MLP Regression for tictac_multi \n")
fold_index = 1
Scores = []
for train_index, test_index in KFold(n_splits=10).split(x_multi):
    print('Fold: ' + str(fold_index) + '\n')
    x_tr_multi, x_ts_multi = x_multi[train_index], x_multi[test_index]
    y_tr_multi, y_ts_multi = y_multi[train_index], y_multi[test_index]

    Reg = MLPRegressor(hidden_layer_sizes=(200, 400, 400, 200), max_iter=500)
    Reg.fit(x_tr_multi, y_tr_multi)

    # Finding optimal threshold for rounding outputs
    val_scores = []
    for tr in range(5, 10):
        yp_val = Reg.predict(x_val_multi)
        yp_val = yp_val > tr*1e-1
        val_scores.append(accuracy_score(y_val_multi, yp_val))
    optimal_tr = (np.argmax(val_scores)+5)*1e-1

    y_pred = Reg.predict(x_ts_multi)
    y_pred = y_pred > optimal_tr

    print("MLP Accuracy: " + str(accuracy_score(y_ts_multi, y_pred)) + '\n')
    Scores.append(accuracy_score(y_ts_multi, y_pred))
    print("MLP Confusion Matrices: " + '\n')
    confmat = multilabel_confusion_matrix(y_ts_multi, y_pred)
    print(confmat)

    fold_index += 1
    print("")
    print("________________________________________________________  \n")

print("MLP Final Result")
print("Overall Multi-Label Accuracy: " + str(np.mean(Scores)) + " +/- " + str(np.std(Scores)) + "\n")
print("________________________________________________________  \n")

# KNN for Multi-Label
# Architecture from Classification
print("KNN Regression for tictac_multi \n")
fold_index = 1
Scores = []
for train_index, test_index in KFold(n_splits=10).split(x_multi):
    print('Fold: ' + str(fold_index) + '\n')
    x_tr_multi, x_ts_multi = x_multi[train_index], x_multi[test_index]
    y_tr_multi, y_ts_multi = y_multi[train_index], y_multi[test_index]

    # Finding optimal parameter
    val_scores = []
    for n in range(1, 10):
        clf_val = KNeighborsRegressor(n_neighbors=n)
        clf_val.fit(x_tr_multi, y_tr_multi)
        yp_val = clf_val.predict(x_val_multi)
        yp_val = yp_val > 0.5
        val_scores.append(accuracy_score(y_val_multi, yp_val))
    optimal_n = (np.argmax(val_scores)+1)

    Reg = KNeighborsRegressor(n_neighbors=optimal_n)
    Reg.fit(x_tr_multi, y_tr_multi)

    # Finding optimal threshold for rounding outputs
    val_scores = []
    for tr in range(5, 10):
        yp_val = Reg.predict(x_val_multi)
        yp_val = yp_val > tr*1e-1
        val_scores.append(accuracy_score(y_val_multi, yp_val))
    optimal_tr = (np.argmax(val_scores)+5)*1e-1

    y_pred = Reg.predict(x_ts_multi)
    y_pred = y_pred > optimal_tr

    print("KNN Accuracy: " + str(accuracy_score(y_ts_multi, y_pred)) + '\n')
    Scores.append(accuracy_score(y_ts_multi, y_pred))
    print("KNN Confusion Matrices: " + '\n')
    confmat = multilabel_confusion_matrix(y_ts_multi, y_pred)
    print(confmat)

    fold_index += 1
    print("")
    print("________________________________________________________  \n")

print("KNN Final Result")
print("Overall Multi-Label Accuracy: " + str(np.mean(Scores)) + " +/- " + str(np.std(Scores)))
