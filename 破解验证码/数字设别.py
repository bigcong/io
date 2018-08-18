import os
import requests

import numpy as np
from PIL import Image

from sklearn import datasets
import tensorflow as tf
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from 破解验证码.Classify_brain import brain
from 破解验证码.GET import GET
import matplotlib.pyplot as plt


def get_train_data():
    xx = []
    yy = []
    path = 'data/'
    lists = os.listdir(path)  # 列出目录的下所有文件和文件夹保存到lists
    for i in lists:
        im = Image.open(path + i)
        im = im.convert("L")  # 转成灰色模式
        data = im.getdata()
        data = np.array(data) / 225  # 转换成矩阵

        yy.append(i.split("_")[0])
        xx.append(np.array(data))
    return np.array(xx), yy


def get_test_data():
    xx = []
    yy = []
    path = 'GET/'
    lists = os.listdir(path)  # 列出目录的下所有文件和文件夹保存到lists
    for i in lists:
        im = Image.open(path + i)
        im = im.convert("L")  # 转成灰色模式
        data = im.getdata()
        data = np.array(data) / 225  # 转换成矩阵

        yy.append(i.split("_")[0])
        xx.append(np.array(data))
    yy = LabelBinarizer().fit_transform(yy)
    return np.array(xx), yy

    # plt.show()


if __name__ == '__main__':
    xx, yy = get_train_data()
    lb = LabelBinarizer();
    yy = lb.fit_transform(yy)
    b = brain(xx, yy)

    for i in range(10):

        try:
            b.train(xx, yy)
            b.test(lb)
        except:
            print("异常")
