# DigitalRecognition

含有小数点的数码管识别，其中包括识别和标定程序

## 如何使用该程序？

#### 数码管的标定：

首先运行*calibration.py*这个文件，这个程序完成的主要功能是为了让识别程序更好、更精确的识别出数码管的数字是什么。假设现在标定的图片为*pic27.jpg*,<img src="D:\PyCharm Community Edition 2019.2.5\Python_code\DigitalRecognition\picture\pic27.jpg" alt="pic27" style="zoom:20%;" />

1、那么观察他的小数点位置是在4个数码管中的第1位（小数点和数字是算在同一个位置）

2、*inv*这个参数是为了背景比数码管颜色要亮的时候用的，默认是黑底白字这种类型。

那么针对这个标定图像，*Digital*这个类写入的参数就是1和*False*。

```python
if __name__ == "__main__":

    img = cv.imread("./picture/pic27.jpg")
    cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
    cv.imshow("image", img)

    dc = Digital(img, decimal_point=1, inv=False)
```

参数设定完成开始运行时，根据需求可以裁剪画面ROI区域（为了去除没用的信息，提高数码管识别准确率），如果选取的图片已经是ROI区域，那就直接关掉这个窗口。
```Python
    cv.waitKey(0)
    dc.binary_pic()
    dc.calibration()

    cv.waitKey(0)
    cv.destroyAllWindows()
```

最后程序会根据参数自动计算出标定配置信息。

- **需要注意是：如果输出的相关配置参数出现负数时，需要重新标定，比如起始坐标，旋转角度出现为负数的时候，就需要检查是不是参数设置错了。**