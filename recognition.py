# -*- coding: utf-8 -*-
# @File : recognition.py
# @Author: Runist
# @Time : 2019/12/20 16:53
# @Software: PyCharm
# @Brief: 数码管识别

import cv2 as cv
import matplotlib.pyplot as plt
import math
import numpy as np


class DigitalRecognition(object):
    def __init__(self, pic, n, roi_x, roi_y, roi_w, roi_h, angle=0.0, segment_width=0, segment_height=0,
                 digital_width=0, digital_height=0, digital_distance=0,
                 start_pos=(0, 0), radix_pos=(0, 0), thresh=None, inv=False):

        self.src = pic

        # 数码管的数量
        self.num = n
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_w
        self.roi_height = roi_h

        # 数码管倾斜角度(数值->角度)
        self.rotation = math.pi/180*angle
        # 数码管 段选长度（短的）
        self.segment_height = segment_height
        # 数码管 段选宽度（长的）
        self.segment_width = segment_width
        # 两个数码管之间的间隔（G段中心点的距离）
        self.digital_distance = digital_distance
        # 段选阈值，如果小于这个数，则认为该段亮起来了，范围（0-25500）
        self.values = 100
        # 单个数码管的长 和 宽，用户可以自己输入，也可以启用标定函数，但是标定的数码管必须为8，2
        self.digital_width, self.digital_height = digital_width, digital_height
        # 识别的起始坐标,元组拆包
        self.start_x, self.start_y = start_pos
        # 小数点的位置(只传入第一个小数点的位置,后面全部用距离偏移计算)
        self.radix_x, self.radix_y = radix_pos
        # 二值化阈值（为了应对不同亮度和远近的识别，如果不设置就是为None，设置了则会将程序原有的大津法覆盖）
        self.thresh_value = thresh
        # 二值化是否要反转黑白
        self.inv_flag = inv

        # 鼠标交互裁剪后感兴趣区域
        self.roi_img = self.src[self.roi_y: self.roi_y+self.roi_height, self.roi_x:self.roi_x+self.roi_width]
        # 黑白二值化后的图像
        self.binary = None
        # 形态学操作后的图像
        self.morphology = None
        # 去识别的图片，可以膨胀，也可以不膨胀
        self.pure = None

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
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        # 竖直方向的膨胀要大一点，比如1 和 7 中间藕断丝连的地方需要大一点
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 8))

        ver_img = cv.dilate(self.binary, vertical_kernel, iterations=iters)
        hor_img = cv.dilate(ver_img, horizontal_kernel, iterations=iters)

        self.morphology = hor_img
        self.pure = self.morphology

        # cv.imshow("hor_img", hor_img)

    def calculate_coordinate(self):
        """
        计算数码管的八段段选坐标，首先要根据倾斜角度计算出偏移量
        偏移量计算公式：tan(a) * 数码管的高度
        之后每一个数码管通过 数码管之间的间距进行偏移计算
        数码管偏移计算： 数码管初始坐标 + 第几个数码管 * 数码管之间的距离
        :return: None
        """
        image = cv.cvtColor(self.pure, cv.COLOR_GRAY2BGR)

        for i in range(self.num):
            # (x1, y1)为左上角的坐标 (x2, y2)为右下角的坐标
            x1 = self.start_x + i * self.digital_distance
            y1 = self.start_y
            x2 = x1 + self.digital_width
            y2 = y1 + self.digital_height
            """
            以下公式都是基于x1, y1, x2, y2坐标进行偏移
            x = x1 + offset1
            y = y1 + offset2
            """

            # A段的偏移量，理论上A段与D段的偏移量是对称的
            offset = int(math.tan(self.rotation) * self.digital_height)

            # A段中心坐标
            a_x = x1 + offset + self.segment_width / 2
            a_y = y1 + self.segment_height
            a_period = (int(a_x), int(a_y))

            # B段
            b_x = x1 + 3 / 4 * offset + self.segment_width
            b_y = y1 + self.digital_height / 4
            b_period = (int(b_x), int(b_y))

            # C段
            c_x = x1 + (1 / 4 * offset) + self.segment_width
            c_y = y1 + 3 / 4 * self.digital_height
            c_period = (int(c_x), int(c_y))

            # D段
            d_x = x1 + self.segment_width / 2
            d_y = y2 - self.segment_height
            d_period = (int(d_x), int(d_y))

            # E段
            e_x = x1 + 1 / 4 * offset
            e_y = y1 + 3 / 4 * self.digital_height
            e_period = (int(e_x), int(e_y))

            # F段
            f_x = x1 + 3 / 4 * offset
            f_y = y1 + 1 / 4 * self.digital_height
            f_period = (int(f_x), int(f_y))

            # G段
            g_x = x1 + offset / 2 + self.segment_width / 2
            g_y = y1 + self.digital_height / 2
            g_period = (int(g_x), int(g_y))

            # 小数点
            h_x, h_y = self.radix_x + i*self.digital_distance, self.radix_y
            h_period = (h_x, h_y)

            # 根据相关坐标判断该段是否亮起，并组成一个列表参数，传入到最后计算数字和小数点的方法中去
            number_status = [self.check(a_period), self.check(b_period), self.check(c_period), self.check(d_period),
                                self.check(e_period), self.check(f_period), self.check(g_period)]

            # 如果小数点输入进来就是（0，0）认为用户没有传入，所以此图中没有小数点
            if self.radix_x == 0 and self.radix_y == 0:
                radix_status = False
            else:
                cv.circle(image, h_period, 3, (255, 100, 100), 1)
                radix_status = self.check(h_period)

            # 计算识别的结果
            print(self.get_num(number_status), end='')
            print(self.get_radix(radix_status), end=' ')

            # 显示 坐标计算结果
            cv.circle(image, a_period, 3, (255, 100, 100), 1)
            cv.circle(image, b_period, 3, (255, 100, 100), 1)
            cv.circle(image, c_period, 3, (255, 100, 100), 1)
            cv.circle(image, d_period, 3, (255, 100, 100), 1)
            cv.circle(image, e_period, 3, (255, 100, 100), 1)
            cv.circle(image, f_period, 3, (255, 100, 100), 1)
            cv.circle(image, g_period, 3, (255, 100, 100), 1)

            cv.imshow("point", image)

    def check(self, point):
        """
        数码管段选阈值检查，低于阈值认为该段没有亮起来，高于该阈值认为该段亮起
        :param point: 段选位置信息
        :return: True or False
        """
        x1 = point[0] - 2
        x2 = point[0] + 2
        y1 = point[1] - 2
        y2 = point[1] + 2

        # 对16个像素点进行求和
        result = np.sum(self.pure[y1: y2, x1: x2], axis=(0, 1))
        if result > self.values:
            return True
        else:
            return False

    @staticmethod
    def get_num(digital_state):
        """
        根据数码管亮灭状态的列表，判断最后的数字
        :param digital_state: 数码管亮灭的状态
        :return: None
        """

        if digital_state == [True, True, True, True, True, True, False]:
            return 0
        elif digital_state == [False, True, True, False, False, False, False]:
            return 1
        elif digital_state == [True, True, False, True, True, False, True]:
            return 2
        elif digital_state == [True, True, True, True, False, False, True]:
            return 3
        elif digital_state == [False, True, True, False, False, True, True]:
            return 4
        elif digital_state == [True, False, True, True, False, True, True]:
            return 5
        elif digital_state == [True, False, True, True, True, True, True]:
            return 6
        elif digital_state == [True, True, True, False, False, False, False]:
            return 7
        elif digital_state == [True, True, True, True, True, True, True]:
            return 8
        elif digital_state == [True, True, True, True, False, True, True]:
            return 9
        elif digital_state == [False, False, False, False, False, False, True]:
            return '-'
        else:
            return '未匹配到数字'

    @staticmethod
    def get_radix(radix_state):
        """
        判断小数点是否存在
        :param radix_state: 小数点段选的亮灭状态
        :return: . 或者 空字符串
        """
        if radix_state:
            return '.'
        else:
            return ''


if __name__ == "__main__":

    img = cv.imread("./picture/pic28.jpg")
    cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
    cv.imshow("image", img)

    dr = DigitalRecognition(img, n=8, roi_x=694, roi_y=386, roi_w=258, roi_h=52, digital_distance=32,
                            digital_width=26, digital_height=38, angle=9.46, start_pos=(0, 8),
                            segment_width=20, inv=False, radix_pos=(0, 0), segment_height=3)
    # 先二值化图片
    dr.binary_pic()
    # 进行形态学操作（膨胀）
    # dr.morphologyEx()
    # 根据相关参数计算识别的数字
    dr.calculate_coordinate()

    cv.waitKey(0)
    cv.destroyAllWindows()
