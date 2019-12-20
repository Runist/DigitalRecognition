# -*- coding: utf-8 -*-
# @File : calibration.py
# @Author: Runist
# @Time : 2019/12/20 16:52
# @Software: PyCharm
# @Brief: 实现对数码管的标定

import cv2 as cv
import matplotlib.pyplot as plt
import math
import numpy as np
import json


class Digital(object):
    def __init__(self, pic, decimal_point=None, thresh=None, inv=False, ver_kernel=(3, 1), hor_kernel=(1, 8)):
        """
        类定义初始化
        :param pic: 输入图像
        :param n: 数码管的个数（数字和小数点算一个）
        :param decimal_point: 小数点在那个位置上（一起算，如果在第2个数码管上，就是2，从1开始算）
        :param thresh: 二值化阈值
        :param ver_kernel: 水平形态学计算的核
        :param hor_kernel: 竖直形态学计算的核
        """
        self.src = pic

        self.point1 = 0
        self.point2 = 0

        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = self.src.shape[0]
        self.roi_height = self.src.shape[1]

        # 标定
        # 数码管倾斜角度(数值->角度)
        self.angle = 0.0
        self.radian = math.pi/180 * self.angle
        # 数码管 段选长度（短的）
        self.segment_height = 0
        # 数码管 段选宽度（长的）
        self.segment_width = 0
        # 两个数码管之间的间隔（G段中心点的距离）
        self.digital_distance = 0
        # 单个数码管的长 和 宽，用户可以自己输入，也可以启用标定函数，但是标定的数码管必须为8，2，5，0，6
        self.digital_width, self.digital_height = 0, 0
        # 识别的起始坐标
        self.start_x, self.start_y = 0, 0
        # 小数点在标定的数码管什么位置(只针对标定程序, 从1开始数)，默认参数是没有的，没有设置位数就是不存在，标定则不考虑小数点
        self.radix_pos = decimal_point
        # 小数点的位置
        self.radix_x, self.radix_y = None, None
        # 二值化阈值（为了应对不同亮度和远近的识别，如果不设置就是为None，设置了则会将程序原有的大津法覆盖）
        self.thresh_value = thresh
        # 二值化是否要反转黑白
        self.inv_flag = inv
        # 形态学算子（为了应对不同大小的识别框，如果相应识别框小一点，算子相应也小一点，反之亦然。默认为（3，1） （1，8））
        self.ver_kernel = ver_kernel
        self.hor_kernel = hor_kernel

        # 轮廓信息
        self.cnts = None
        # 含有小数点的数码管坐标信息
        self.location = []
        # 后续程序自动计算这两个标定的数码管位置
        self.standard1 = None
        self.standard2 = None

        # ROI区域
        self.roi_img = self.src
        # 黑白二值化后的图像
        self.binary = None
        # 形态学操作后的图像
        self.morphology = None
        # 去标定的图片，可以膨胀，也可以不膨胀
        self.pure = None

        # 对象生成后，直接进行裁剪
        cv.setMouseCallback('image', self.get_roi)

    def get_roi(self, event, x, y, flags, param):
        """
        回调函数，图像初步截取，找到需要进行识别的区域
        :param event: 鼠标事件类型
        :param x: 鼠标的x坐标
        :param y: 鼠标的y坐标
        :param flags: 时间标志位
        :param param: 在C++中为void空指针类型
        :return: None
        """
        img1 = self.src.copy()
        img2 = img.copy()

        if event == cv.EVENT_LBUTTONDOWN:  # 左键点击
            self.point1 = (x, y)
            cv.circle(img2, self.point1, 10, (0, 255, 0), 2)
            cv.imshow('image', img2)

        elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
            cv.rectangle(img2, self.point1, (x, y), (255, 0, 0), 2)
            cv.imshow('image', img2)

        elif event == cv.EVENT_LBUTTONUP:  # 左键释放
            self.point2 = (x, y)
            cv.rectangle(img2, self.point1, self.point2, (0, 0, 255), 2)
            cv.imshow('image', img2)

            self.roi_x = min(self.point1[0], self.point2[0])
            self.roi_y = min(self.point1[1], self.point2[1])
            self.roi_width = abs(self.point1[0] - self.point2[0])
            self.roi_height = abs(self.point1[1] - self.point2[1])

            self.roi_img = img1[self.roi_y: self.roi_y + self.roi_height, self.roi_x: self.roi_x + self.roi_width]
            print("所要识别的区域起始坐标为:%d %d," % self.point1, end='')
            print("结束坐标为:%d %d" % self.point2)
            self.roi_x, self.roi_y = self.point1
            self.roi_width, self.roi_height = self.point2[0] - self.point1[0], self.point2[1] - self.point1[1]
            cv.imshow('cut_img', self.roi_img)

    def binary_pic(self):
        """
        图像二值化
        :return: None
        """
        # 图片采用全局二值化， OTSU算法自动计算最佳阈值
        gray = cv.cvtColor(self.roi_img, cv.COLOR_BGR2GRAY)

        if self.inv_flag:
            method = cv.THRESH_BINARY_INV
        else:
            method = cv.THRESH_BINARY

        if self.thresh_value is None:
            ret, self.binary = cv.threshold(gray, 0, 255, method | cv.THRESH_OTSU)
        else:
            ret, self.binary = cv.threshold(gray, self.thresh_value, 255, method)

        self.pure = self.binary
        cv.imshow("binary_img", self.binary)

    def morphologyEx(self, iters=1):
        """
        图像形态学操作：主要是进行一个膨胀，但是后期要根据实际图像大小进行对核的大小改动
        :param iters: 形态学操作的次数
        :return: None
        """
        # 分为水平方向和竖直方向的形态学的核
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, self.ver_kernel)
        # 竖直方向的膨胀要大一点，比如1 和 7 中间藕断丝连的地方需要大一点
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, self.hor_kernel)

        ver_img = cv.dilate(self.binary, vertical_kernel, iterations=iters)
        hor_img = cv.dilate(ver_img, horizontal_kernel, iterations=iters)

        self.morphology = hor_img

        self.pure = self.morphology
        cv.imshow("morphology_img", hor_img)

    def calibration(self):
        """
        标定程序主要部分，步骤如下：
        建立外包矩形，根据返回的坐标，计算各种参数
        :return: None
        """
        self.cnts, _ = cv.findContours(self.pure, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.bgr_img = cv.cvtColor(self.pure, cv.COLOR_GRAY2BGR)

        for cnt in self.cnts:
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(self.bgr_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.location.append([x, y, w, h])

        # 轮廓信息（可能小数点是一个框，数字是一个框）
        self.location = sorted(self.location, key=(lambda x: x[0]))

        # 计算步骤：
        # 1、通过轮廓信息，先得出“标准的”数码管
        # 2、计算数码管之间的间距
        # 3、计算小数点位置
        # 4、先得出单个数码管显示区域的 长 宽
        # 5、计算旋转角度
        # 6、计算段选长 宽
        # 7、计算第一个数码管的起始坐标（左上角）
        self.get_standard_digital()
        self.get_digital_distance()
        self.get_digital_radix_point()
        self.get_digital_width_height()
        self.get_digital_rotation()
        self.get_digital_segment_width_height()
        self.get_first_digital_pos()

        # 验证段选长度
        cv.line(self.bgr_img,
                (self.start_x, self.start_y + self.digital_height - self.segment_height),
                (self.start_x+self.segment_width, self.start_y+self.digital_height-self.segment_height), (0, 255, 255), 2)
        # 验证数码管之间的距离
        cv.line(self.bgr_img,
                (self.start_x + self.digital_width//2, self.start_y + self.digital_height//2),
                (self.start_x + self.digital_width//2 + self.digital_distance,
                 self.start_y + self.digital_height//2), (255, 255, 0), 2)
        # 验证旋转角度
        offset = int(math.tan(self.radian) * self.digital_height)
        cv.line(self.bgr_img,
                (self.start_x + self.segment_height, self.start_y + self.digital_height - self.segment_height),
                (self.start_x + self.segment_height + offset, self.start_y + self.segment_height), (0, 255, 0), 2)

        # 验证第一个数码管起始坐标
        cv.circle(self.bgr_img, (self.start_x, self.start_y), 2, (0, 255, 0), -1)

        cv.imshow("out", self.bgr_img)

        print('数码管之间距离为：%d' % self.digital_distance)
        if self.radix_x or self.radix_y:
            # 验证小数点位置
            cv.circle(self.bgr_img, (self.radix_x, self.radix_y), 2, (0, 255, 0), -1)
            print('数码管第一个小数点的位置:(%d, %d)' % (self.radix_x, self.radix_y))
        else:
            print('数码管没有亮起小数点')

        print('单个数码管的长度与高度：%d, %d' % (self.digital_width, self.digital_height))
        print('数码管的旋转角度为%.2f' % self.angle)
        print('数码管起始坐标为：(%d, %d)' % (self.start_x, self.start_y))
        print('单个数码管的段选长度：%d' % self.segment_width)
        print('二值化反转flag：%s' % self.inv_flag)

    def get_standard_digital(self):
        """
        获取图片中程序认为“标准”的数码管，用来作为标定参数
        选取面积最大的两个数码管，作为“标准”数码管，其中：
        如果有8或0会提高标定精度
        :return: None
        """
        loc = []
        for i in range(len(self.location)):
            size = self.location[i][2] * self.location[i][3]
            loc.append([i, size])

        loc = sorted(loc, key=lambda x: x[1], reverse=True)
        self.standard1 = loc[0][0]
        self.standard2 = loc[1][0]

    def get_first_digital_pos(self):
        """
        计算数码管起始坐标
        :return: None
        """
        # 计算数码管起始坐标
        x, y = self.location[self.standard1][:2]

        # 如果面积最大的数码管位置比小数点的位置要大，算坐标x坐标距离的时候，要先-1再乘上距离
        if self.radix_pos is None or self.standard1 < self.radix_pos:
            self.start_x = x - self.digital_distance * self.standard1
        else:
            self.start_x = x - self.digital_distance * (self.standard1 - 1)
        self.start_y = y
        if self.start_x < 0 or self.start_y < 0:
            print('数码管起始坐标小于0，ROI区域过小，请重新标定，避免识别错误')

    def get_digital_width_height(self):
        """
        计算单个数码管长度、高度
        :return: None
        """
        # 计算单个数码管长度、高度
        self.digital_width, self.digital_height = self.location[self.standard1][2:]

    def get_digital_distance(self):
        """
        计算数码管之间的间距，利用 小数点和数字坐标信息的 高度差异，先提取出数码管间的距离
        :return: None
        """
        l1 = self.standard1
        l2 = self.standard2

        # 理论上l1 应该大于 l2，但为了提高鲁棒性，使用绝对值
        if self.radix_pos is None or (l1 - self.radix_pos > 0 and l2 - self.radix_pos > 0) or \
            (l1 - self.radix_pos < 0 and l2 - self.radix_pos < 0):
            self.digital_distance = abs(self.location[l2][0] - self.location[l1][0]) // abs(l1 - l2)
        else:
            # 认为小数点在这两个标定数码管之间
            # 那么计算距离的时候 - 1 即可
            self.digital_distance = abs(self.location[l2][0] - self.location[l1][0]) // (abs(l1 - l2) - 1)

    def get_digital_radix_point(self):
        """
        计算小数点的位置(简单识别出位置),要把小数点位置移动到第一个点,也就是数码管第二个位置,如12.3- > 1.23,方便识别程序
        最后输出的 小数点位置就是第一个小数点的位置
        移动距离计算公式: 小数点位置/2 * 数码管的距离
        :return: None
        """
        # 如果有小数点，要先把小数点先擦除
        if self.radix_pos is not None:
            # 计算小数点的位置(简单识别出位置),要把小数点位置移动到第一个点,也就是数码管第二个位置,如12.3- > 1.23,方便识别程序
            # 最后输出的 小数点位置就是第一个小数点的位置
            # 移动距离计算公式: 小数点位置/2 * 数码管的距离
            x, y, w, h = self.location[self.radix_pos]
            self.radix_x, self.radix_y = int((x + w/2) - (self.radix_pos//2)*self.digital_distance), int((y + h/2))

    def get_digital_segment_width_height(self):
        """
        计算数码管的段选 长度 以及 宽度（非必要，会在识别程序计算出来）
        :return: None
        """
        # 同样，计算偏移量、数码管段选长度要放在角度之后，有了角度才能计算偏移
        offset = int(math.tan(self.radian) * self.digital_height)
        self.segment_width = int(self.digital_width - offset)
        # 计算数码管段选宽度(未计算出)
        self.segment_height = 0

    def get_digital_rotation(self):
        """
        计算数码管的平均偏转角度
        不能标定除了8 与 0 之外的数字，否则其他数字
        数码管灭灯状态会影响角度的识别
        :return: None, 作为类属性存储进去
        """
        angle = [cv.minAreaRect(self.cnts[self.standard1])[2], cv.minAreaRect(self.cnts[self.standard2])[2]]

        # 求各个数码管偏转的平均值，然后计算其转换为角度,如果angle列表长度为0,代表返回为全0,python中0=False,None所以angle长度为0
        if len(angle) == 0:
            self.angle = 0.0
        else:
            self.angle = 90 + (sum(angle) / len(angle))

        # 转换为tan函数需要的弧度制
        self.radian = math.pi/180 * self.angle


if __name__ == "__main__":

    img = cv.imread("./picture/test6.png")
    cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
    cv.imshow("image", img)

    dc = Digital(img, decimal_point=None, inv=True)

    # 实现正常退出
    cv.waitKey(0)
    dc.binary_pic()
    # dc.morphologyEx()
    dc.calibration()
    # dc.generate_json()

    cv.waitKey(0)
    cv.destroyAllWindows()
