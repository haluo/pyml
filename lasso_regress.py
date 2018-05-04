# coding: utf-8
'''
@author: jessezhu
@file: lasso_regress.py
@time: 2018/5/4 上午10:31
@desc:

'''

import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import numpy as np


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))