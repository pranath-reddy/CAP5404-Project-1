'''
CAP5404 Deep Learning for Computer Graphics
Project Part-1
Author: Pranath Reddy Kumbam
UFID: 8512-0977

- Train a linear regression model on the Tic Tac Toe multi-label dataset using normal equations
'''

# Import libraries
from sklearn.utils import shuffle
import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pickle

# Linear Regression Function
def LinReg(x, y):
    # Returns weights
    # Based on Normal Equation W = (((X^T * X)^-1)*X^T)*Y (Eq1 in Report)
    w = inv(np.dot(np.transpose(x),x))
    w = np.dot(w,np.transpose(x))
    w = np.dot(w,y)
    return w

# Load Data
data_multi = np.loadtxt('./Data/tictac_multi.txt')[:, :9]
labels_multi = np.loadtxt('./Data/tictac_multi.txt')[:, 9:]

# Split the data into validation set
x_multi, x_val_multi, y_multi, y_val_multi = train_test_split(data_multi, labels_multi, test_size=0.05)

# Shuffle the data
x_multi, y_multi = shuffle(x_multi, y_multi, random_state=0)
x_val_multi, y_val_multi = shuffle(x_val_multi, y_val_multi, random_state=0)
y_val_multi = y_val_multi.transpose()

# Linear Regression for Multi-Label
print("Regression for tictac_multi \n")
fold_index = 1
Scores = []
Match_Scores = []
for train_index, test_index in KFold(n_splits=10).split(x_multi):
    print('Fold: ' + str(fold_index) + '\n')
    y_predictions = []
    for m, y in enumerate(y_multi.transpose()):
        x_tr_multi, x_ts_multi = x_multi[train_index], x_multi[test_index]
        y_tr_multi, y_ts_multi = y.transpose()[train_index], y.transpose()[test_index]
        y_val = y_val_multi[m].transpose()

        # Fitting and testing the model
        w = LinReg(x_tr_multi, y_tr_multi)
        yp_val = np.dot(x_ts_multi, w)
        y_predictions.append(yp_val)

    y_predictions = np.asarray(y_predictions).transpose()
    y_pred = np.zeros(y_predictions.shape)
    for i in range(y_pred.shape[0]):
        y_pred[i][np.argmax(y_predictions[i])] = 1
    print(accuracy_score(y_multi[test_index], y_pred))
    Scores.append(accuracy_score(y_multi[test_index], y_pred))

    Matches = 0
    for i in range(y_pred.shape[0]):
        Matches += 1 if y_multi[test_index][i][np.argmax(y_pred[i])] == 1 else 0
    print(Matches/y_pred.shape[0])
    Match_Scores.append(Matches/y_pred.shape[0])

    fold_index += 1
    print("")
    print("________________________________________________________  \n")

# Print the overall scores
print("Results")
print("Overall Multi-Label Accuracy: " + str(np.mean(Scores)) + " +/- " + str(np.std(Scores)))
print("Accuracy Based On Correctly Predicted Optimal Moves: " + str(np.mean(Match_Scores)) + " +/- " + str(np.std(Match_Scores)))
