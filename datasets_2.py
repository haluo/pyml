# coding: utf-8
'''
@author: jessezhu
@file: datasets_1.py
@time: 2018/5/2 下午4:22
@desc:

'''

import matplotlib.pyplot as plt
import mglearn



X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()