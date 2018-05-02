# coding: utf-8
'''
@author: jessezhu
@file: datasets_1.py
@time: 2018/5/2 下午4:22
@desc:

'''

import matplotlib.pyplot as plt
import mglearn



# 生成数据集
X, y = mglearn.datasets.make_forge()
# 数据集绘图
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
print("X : {}".format(X))
print("y.shape: {}".format(y.shape))
print("y : {}".format(y))
plt.show()
