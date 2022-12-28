'''
CAP5404 Deep Learning for Computer Graphics
Project Part-1
Author: Pranath Reddy Kumbam
UFID: 8512-0977

- Train Random Forest model to deploy
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

# Balance the Data based on Class with lowest No Of samples
data_win = np.asarray([sample for sample in data if sample[-1] == 1])
data_loss = np.asarray([sample for sample in data if sample[-1] == 0])
data_draw = np.asarray([sample for sample in data if sample[-1] == 2])
data2 = (np.concatenate((data_win[:data_draw.shape[0]], data_loss[:data_draw.shape[0]], data_draw)))
np.random.shuffle(data2)

# Training and exporting on Balanced Data
x, y = shuffle(data2[:,:-1], data2[:,-1], random_state=0)
print(x.shape)
clf_RF = RandomForestClassifier(max_depth=40, random_state=0)
clf_RF.fit(x, y)
pickle.dump(clf_RF, open("RF_Connect4_Balanced.sav", 'wb'))

yp_RF = clf_RF.predict(x)
acc_RF = accuracy_score(y, yp_RF)
print("RF Training Accuracy: " + str(acc_RF) + '\n')

# Training and exporting on Unbalanced Data
x, y = shuffle(data[:,:-1], data[:,-1], random_state=0)
print(x.shape)
clf_RF = RandomForestClassifier(max_depth=40, random_state=0)
clf_RF.fit(x, y)
pickle.dump(clf_RF, open("RF_Connect4_Unbalanced.sav", 'wb'))

yp_RF = clf_RF.predict(x)
acc_RF = accuracy_score(y, yp_RF)
print("RF Training Accuracy: " + str(acc_RF) + '\n')
