import collections

import numpy as np
import tensorflow as tf

import requests
from io import BytesIO
import base64

from PIL import Image
import matplotlib.pyplot as plt;


class GET(object):
    # input_image_shape  图片的形状比如：长*宽*高（通道数)，get_image_url 下载图片的路径
    def __init__(self,
                 get_image_url='http://59.110.157.9/polarisex/security/getCode',
                 verify_url='',
                 code_size=5
                 ):
        self.get_image_url = get_image_url
        self.code_size = code_size
        self.getImage()
        self.dms = []

        # self.iamge2imbw()

    def getImage(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}
        json = requests.post(self.get_image_url, headers=headers).json()

        IMGCode = json['attachment']['IMGCode']
        codeUUID = json['attachment']['codeUUID']
        imdate = base64.b64decode(IMGCode)
        self.codeUUID = codeUUID
        self.im = Image.open(BytesIO(imdate))

    def plot(self):
        codeUUID, im = self.codeUUID, self.im.convert("L")
        w, h = im.size
        print(w, h)
        plt.figure()
        crop_w, crop_h = w // self.code_size, h

        for j in range(self.code_size):
            box = (crop_w * j, 00, (1 + j) * crop_w, crop_h)
            print(box)
            plt.subplot(2, 3, j + 1)
            dm = im.crop(box)
            dm = self.iamge2imbw(dm)
            plt.imshow(np.array(dm.getdata()).reshape(crop_h, crop_w) / 255)
        plt.subplot(2, 3, self.code_size + 1)
        plt.imshow(np.array(im.getdata()).reshape(h, w) / 255)
        plt.ion()
        plt.show()

    def spit(self):
        codeUUID, im = self.codeUUID, self.im.convert("L")
        w, h = im.size

        crop_w, crop_h = w // self.code_size, h
        for j in range(self.code_size):
            box = (crop_w * j, 00, (1 + j) * crop_w, crop_h)
            dm = im.crop(box)
            dm = self.iamge2imbw(dm)
            dm.save("get/" + str(j) + "_" + self.codeUUID + ".png")
            self.dms.append(np.array(dm.getdata())/ 255)
        self.dms = np.array(self.dms)

    def iamge2imbw(self, img, inde=1):
        """传入image对象进行灰度、二值处理"""
        img = img.convert("L")  # 转灰度
        pixdata = img.load()
        w, h = img.size
        # 遍历所有像素，大于阈值的为黑色
        total = 0
        a = 0;
        b = 1;

        gg = []
        for y in range(h):
            for x in range(w):
                gg.append(pixdata[x, y])

        g = collections.Counter(gg)

        threshold = list(g.most_common())[inde][0]

        for y in range(h):
            for x in range(w):
                if pixdata[x, y] != threshold:
                    pixdata[x, y] = 255
                    a = a + 1

                else:
                    pixdata[x, y] = 0
                    b = b + 1;

        if (b / a) < 0.05:
            print("阀值为：" + str(threshold));
            print(g)

        return img

    def verify_code(self):
        pass


if __name__ == '__main__':
    x = GET()
    x.plot()

    print(x.dms)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 提取变量
        saver.restore(sess, "logs/save_net.ckpt")
