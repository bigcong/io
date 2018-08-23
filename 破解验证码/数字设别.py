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

from 破解验证码.cnn_brain import cnn_brain


def get_train_data():
    xx = []
    yy = []
    path = '//Users/cc/cc/io/破解验证码/data/'
    lists = os.listdir(path)  # 列出目录的下所有文件和文件夹保存到lists
    for i in lists:
        im = Image.open(path + i)
        data = im.getdata()
        data = np.array(data) / 255.0  # 转换成矩阵
        if (len(i.split("_")) == 2):
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
        data = np.array(data) / 225.0  # 转换成矩阵

        yy.append(i.split("_")[0])
        xx.append(np.array(data))

    yy = LabelBinarizer().fit_transform(yy)
    return np.array(xx), yy

    # plt.show()
