# -*- coding: utf-8 -*-
# @File : sift.py
# @Author: Runist
# @Time : 2020/1/8 13:38
# @Software: PyCharm
# @Brief: 图像变换


import numpy as np
import cv2 as cv


img = cv.imread('./picture/pic30.jpg')

rows, cols = img.shape[:2]

# M = cv.getRotationMatrix2D((cols/2, rows/2), 4.5, 1.2)
# new_img = cv.warpAffine(img, M, (2*cols, rows))

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# sift = cv.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)
# img = cv.drawKeypoints(gray, kp, img)


cv.imshow("src", img)
cv.waitKey()
