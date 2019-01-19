# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 09:45:09 2019

@author: One
"""

from PIL import Image
from keras.models import load_model
import cv2
characters="京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])



model = load_model("resnet34_model.h5")
print("导入模型完成")
print("读取图片")
pic = Image.open("C:/Users/One/Desktop/y.jpg")
pic.show()
#这里换两种方式是因为两种方式显示的通道顺序不同

img = cv2.imread("C:/Users/One/Desktop/y.jpg")
img=img[np.newaxis,:,:,:]#图片是三维的但是训练时是转换成4维了所以需要增加一个维度
predict = model.predict(img)

print("车牌号为：",decode(predict))






