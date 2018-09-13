import tensorflow as tf
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('./data/data.csv')

cols = len(data.columns)

m, _ = data.shape

permutation = np.random.permutation(m)
data = data.iloc[permutation]

# split into X and Y
X = data.iloc[:, :cols-1]
Y = data.iloc[:, cols-1:]

# split into training and test sets
split = int(m * 0.8)
X_train, X_test = X.iloc[:split, :], X.iloc[split:, :]
Y_train, Y_test = Y.iloc[:split, :], Y.iloc[split:, :] 

X_train, X_test = np.array(X_train), np.array(X_test)
Y_train, Y_test = np.array(Y_train), np.array(Y_test)
Y_train, Y_test = Y_train.reshape((Y_train.shape[0])), Y_test.reshape((Y_test.shape[0]))

data_train = xgb.DMatrix(X_train, label=Y_train)
data_test = xgb.DMatrix(X_test, label=Y_train)

model = XGBClassifier()
model.fit(data_train)




