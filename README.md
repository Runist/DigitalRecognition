# DigitalRecognition

含有小数点的数码管识别，其中包括识别和标定程序

### 如何使用该程序？

首先运行calibration.py这个文件，这个程序完成的主要功能是为了让识别程序更好、更精确的识别出数码管的数字是什么。假设现在标定的图片为test6.png,<img src="D:\PyCharm Community Edition 2019.2.5\Python_code\DigitalRecognition\picture\test6.png" alt="test6" style="zoom:10%;" />

```python
if __name__ == "__main__":

    img = cv.imread("./picture/test6.png")
    cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
    cv.imshow("image", img)

    dc = Digital(img, decimal_point=None, inv=True)

    # 实现正常退出
    cv.waitKey(0)
    dc.binary_pic()
    dc.calibration()

    cv.waitKey(0)
    cv.destroyAllWindows()

```

