---
title: OpenCV
date: 2020-05-24
tags: 计算机视觉
---

### cv2.imread()

`cv2.imread(filename[, flags]) `

第二个参数是要告诉函数应该如何读取这幅图片。

• `cv2.IMREAD_COLOR`：读入一副彩色图像。图像的透明度会被忽略，这是默认参数。

• `cv2.IMREAD_GRAYSCALE`：以灰度模式读入图像

返回（高度，宽度，通道数）的np数组，可通过.shape查看

<!-- more -->



### cv2.resize()

```python
cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
```

参数：

| 参数          | 描述                       |
| ------------- | -------------------------- |
| src           | 【必需】原图像             |
| dsize         | 【必需】输出图像所需大小   |
| fx            | 【可选】沿水平轴的比例因子 |
| fy            | 【可选】沿垂直轴的比例因子 |
| interpolation | 【可选】插值方式           |

其中插值方式有很多种：

| cv.INTER_NEAREST | 最近邻插值                                                   |
| ---------------- | ------------------------------------------------------------ |
| cv.INTER_LINEAR  | 双线性插值                                                   |
| cv.INTER_CUBIC   | 双线性插值                                                   |
| cv.INTER_AREA    | 使用像素区域关系重新采样。它可能是图像抽取的首选方法，因为它可以提供无莫尔条纹的结果。但是当图像被缩放时，它类似于INTER_NEAREST方法。 |

通常的，缩小使用cv.INTER_AREA，放缩使用cv.INTER_CUBIC(较慢)和cv.INTER_LINEAR(较快效果也不错)。
默认情况下，所有的放缩都使用cv.INTER_LINEAR。