'''
CAP5404 Deep Learning for Computer Graphics
Project Part-1
Author: Pranath Reddy Kumbam
UFID: 8512-0977

- Train Classifiers on the Tic Tac Toe Data(final, single) (KNN, SVM(Linear, RBF), MLP)
'''

# Import libraries
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
data_final = np.loadtxt('./Data/tictac_final.txt')[:, :-1]
labels_final = np.loadtxt('./Data/tictac_final.txt')[:, -1:].reshape(-1)
data_single = np.loadtxt('./Data/tictac_single.txt')[:, :-1]
labels_single = np.loadtxt('./Data/tictac_single.txt')[:, -1:].reshape(-1)

# Split the data into validation set
x_final, x_val_final, y_final, y_val_final = train_test_split(data_final, labels_final, test_size=0.05)
x_single, x_val_single, y_single, y_val_single = train_test_split(data_single, labels_single, test_size=0.05)

# Shuffle the data
x_final, y_final = shuffle(x_final, y_final, random_state=0)
x_val_final, y_val_final = shuffle(x_val_final, y_val_final, random_state=0)

x_single, y_single = shuffle(x_single, y_single, random_state=0)
x_val_single, y_val_single = shuffle(x_val_single, y_val_single, random_state=0)

# Binary Classification
print("Binary Classification for tictac_final \n")
KN_Scores = []
SV_Scores = []
SV2_Scores = []
MLP_Scores = []
fold_index = 1
for train_index, test_index in KFold(n_splits=10).split(x_final):
    print('Fold: ' + str(fold_index) + '\n')
    x_tr_final, x_ts_final = x_final[train_index], x_final[test_index]
    y_tr_final, y_ts_final = y_final[train_index], y_final[test_index]

    # KNN
    # Finding optimal parameter
    val_scores = []
    for n in range(1, 10):
        clf_val = KNeighborsClassifier(n_neighbors=n)
        clf_val.fit(x_tr_final, y_tr_final)
        yp_val = clf_val.predict(x_val_final)
        val_scores.append(accuracy_score(y_val_final, yp_val))
    optimal_n = (np.argmax(val_scores)+1)

    # Fitting and testing the model
    clf_KN = KNeighborsClassifier(n_neighbors=optimal_n)
    clf_KN.fit(x_tr_final, y_tr_final)
    yp_KN = clf_KN.predict(x_ts_final)
    acc_KN = accuracy_score(y_ts_final, yp_KN)
    print("KNN Accuracy: " + str(acc_KN) + '\n')
    KN_Scores.append(acc_KN)
    print("KNN Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts_final, yp_KN, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confmat, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('./Results/Maps/KNN_Binary_Confmat_' + str(fold_index) + '.png', format='png', dpi=300)

    # SVM Linear
    # Fitting and testing the model
    clf_SV = LinearSVC(tol=1e-5)
    clf_SV.fit(x_tr_final, y_tr_final)
    yp_SV = clf_SV.predict(x_ts_final)
    acc_SV = accuracy_score(y_ts_final, yp_SV)
    print("SVM Linear Accuracy: " + str(acc_SV) + '\n')
    SV_Scores.append(acc_SV)
    print("SVM Linear Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts_final, yp_SV, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confmat, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('./Results/Maps/SVM_Binary_Confmat_' + str(fold_index) + '.png', format='png', dpi=300)

    # SVM RBF
    # Fitting and testing the model
    clf_SV = SVC(tol=1e-5, kernel='rbf')
    clf_SV.fit(x_tr_final, y_tr_final)
    yp_SV = clf_SV.predict(x_ts_final)
    acc_SV = accuracy_score(y_ts_final, yp_SV)
    print("SVM RBF Accuracy: " + str(acc_SV) + '\n')
    SV2_Scores.append(acc_SV)
    print("SVM RBF Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts_final, yp_SV, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confmat, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('./Results/Maps/SVM_RBF_Binary_Confmat_' + str(fold_index) + '.png', format='png', dpi=300)

    # MLP
    # After testing 1,2,3,4,5 hidden layers, found 3 to work the best
    # Finding the best layer width:
    '''
    val_scores = []
    for n in range(1, 500, 50):
        clf_val = MLPClassifier(hidden_layer_sizes=(n, n*2, n), max_iter=500)
        clf_val.fit(x_tr_final, y_tr_final)
        yp_val = clf_val.predict(x_val_final)
        val_scores.append(accuracy_score(y_val_final, yp_val))
    optimal_n = (np.argmax(val_scores)+1)*50
    print('optimal neurons: ' + str(optimal_n))
    '''

    # Fitting and testing the model
    # Is model capacity > no of data points?
    optimal_n = 200
    clf_MLP = MLPClassifier(hidden_layer_sizes=(optimal_n, optimal_n*2, optimal_n), max_iter=500)
    clf_MLP.fit(x_tr_final, y_tr_final)
    yp_MLP = clf_MLP.predict(x_ts_final)
    acc_MLP = accuracy_score(y_ts_final, yp_MLP)
    print("MLP Accuracy: " + str(acc_MLP) + '\n')
    MLP_Scores.append(acc_MLP)
    print("MLP Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts_final, yp_MLP, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confmat, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('./Results/Maps/MLP_Binary_Confmat_' + str(fold_index) + '.png', format='png', dpi=300)

    print("________________________________________________________  \n")

    fold_index += 1

# Print the overall scores
print("Results")
print("KNN Accuracy: " + str(np.mean(KN_Scores)) + " +/- " + str(np.std(KN_Scores)))
print("SVM (Linear) Accuracy: " + str(np.mean(SV_Scores)) + " +/- " + str(np.std(SV_Scores)))
print("SVM (RBF) Accuracy: " + str(np.mean(SV2_Scores)) + " +/- " + str(np.std(SV2_Scores)))
print("MLP Accuracy: " + str(np.mean(MLP_Scores)) + " +/- " + str(np.std(MLP_Scores)))
print("________________________________________________________  \n")

# Multi-Class Classification
print("Multi-Class Classification for tictac_single \n")
KN_Scores = []
SV_Scores = []
SV2_Scores = []
MLP_Scores = []
fold_index = 1
for train_index, test_index in KFold(n_splits=10).split(x_single):
    print('Fold: ' + str(fold_index) + '\n')
    x_tr_single, x_ts_single = x_single[train_index], x_single[test_index]
    y_tr_single, y_ts_single = y_single[train_index], y_single[test_index]

    # KNN
    # Finding optimal parameter
    val_scores = []
    for n in range(1, 10):
        clf_val = KNeighborsClassifier(n_neighbors=n)
        clf_val.fit(x_tr_single, y_tr_single)
        yp_val = clf_val.predict(x_val_single)
        val_scores.append(accuracy_score(y_val_single, yp_val))
    optimal_n = (np.argmax(val_scores)+1)

    # Fitting and testing the model
    clf_KN = KNeighborsClassifier(n_neighbors=optimal_n)
    clf_KN.fit(x_tr_single, y_tr_single)
    yp_KN = clf_KN.predict(x_ts_single)
    acc_KN = accuracy_score(y_ts_single, yp_KN)
    print("KNN Accuracy: " + str(acc_KN) + '\n')
    KN_Scores.append(acc_KN)
    print("KNN Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts_single, yp_KN, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confmat, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('./Results/Maps/KNN_Multi_Confmat_' + str(fold_index) + '.png', format='png', dpi=300)

    # SVM Linear
    # Fitting and testing the model
    clf_SV = LinearSVC(tol=1e-5)
    clf_SV.fit(x_tr_single, y_tr_single)
    yp_SV = clf_SV.predict(x_ts_single)
    acc_SV = accuracy_score(y_ts_single, yp_SV)
    print("SVM Linear Accuracy: " + str(acc_SV) + '\n')
    SV_Scores.append(acc_SV)
    print("SVM Linear Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts_single, yp_SV, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confmat, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('./Results/Maps/SVM_Multi_Confmat_' + str(fold_index) + '.png', format='png', dpi=300)

    # SVM RBF
    # Fitting and testing the model
    clf_SV = SVC(tol=1e-5, kernel='rbf')
    clf_SV.fit(x_tr_single, y_tr_single)
    yp_SV = clf_SV.predict(x_ts_single)
    acc_SV = accuracy_score(y_ts_single, yp_SV)
    print("SVM RBF Accuracy: " + str(acc_SV) + '\n')
    SV2_Scores.append(acc_SV)
    print("SVM RBF Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts_single, yp_SV, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confmat, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('./Results/Maps/SVM_RBF_Multi_Confmat_' + str(fold_index) + '.png', format='png', dpi=300)

    # MLP
    # Model was not converging with one hidden layer
    # After testing 1,2,3,4,5 hidden layers, found 4 to work the best
    # Finding the best layer width:
    '''
    val_scores = []
    for n in range(1, 500, 50):
        clf_val = MLPClassifier(hidden_layer_sizes=(n, n*2, n*2, n), max_iter=500)
        clf_val.fit(x_tr_single, y_tr_single)
        yp_val = clf_val.predict(x_val_single)
        val_scores.append(accuracy_score(y_val_single, yp_val))
    optimal_n = (np.argmax(val_scores)+1)*50
    print('optimal neurons: ' + str(optimal_n))
    '''

    # Fitting and testing the model
    # Is model capacity > no of data points?
    optimal_n = 200
    clf_MLP = MLPClassifier(hidden_layer_sizes=(optimal_n, optimal_n*2, optimal_n*2, optimal_n), max_iter=500)
    clf_MLP.fit(x_tr_single, y_tr_single)
    yp_MLP = clf_MLP.predict(x_ts_single)
    acc_MLP = accuracy_score(y_ts_single, yp_MLP)
    print("MLP Accuracy: " + str(acc_MLP) + '\n')
    MLP_Scores.append(acc_MLP)
    print("MLP Confusion Matrix: " + '\n')
    confmat = confusion_matrix(y_ts_single, yp_MLP, normalize='true')
    for row in confmat:
        print(*row, sep="\t")
    print("")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confmat, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('./Results/Maps/MLP_Multi_Confmat_' + str(fold_index) + '.png', format='png', dpi=300)

    print("________________________________________________________  \n")

    fold_index += 1

# Print the overall scores
print("Results: ")
print("KNN Accuracy: " + str(np.mean(KN_Scores)) + " +/- " + str(np.std(KN_Scores)))
print("SVM (Linear) Accuracy: " + str(np.mean(SV_Scores)) + " +/- " + str(np.std(SV_Scores)))
print("SVM (RBF) Accuracy: " + str(np.mean(SV2_Scores)) + " +/- " + str(np.std(SV2_Scores)))
print("MLP Accuracy: " + str(np.mean(MLP_Scores)) + " +/- " + str(np.std(MLP_Scores)))
