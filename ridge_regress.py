# coding: utf-8
'''
@author: jessezhu
@file: ridge_regress.py
@time: 2018/5/3 上午11:03
@desc:

'''
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))