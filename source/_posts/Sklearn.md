---
title: Sklearn
date: 2020-06-03
tags: 机器学习
---

##  confusion_matrix

1. 概念

   混淆矩阵是机器学习中总结分类模型预测结果的情形分析表，以矩阵形式将数据集中的记录按照真实的类别与分类模型作出的分类判断两个标准进行汇总。

   <img src="https://img-blog.csdn.net/20170814211735042?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbTBfMzgwNjE5Mjc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="这里写图片描述" style="zoom:50%;" />

   灰色部分是与真实分类与预测分类结果相同，蓝色是分类错误的。

<!-- more -->

2. confusion_matrix函数的使用

   ``` python
   sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
   ```

   | 参数          | 性质                                     |
   | ------------- | ---------------------------------------- |
   | y_true        | 样本真实分类结果                         |
   | y_pred        | 样本预测分类结果                         |
   | labels        | 是所给出的类别，通过这个可对类别进行选择 |
   | sample_weight | 样本权重                                 |

   

## classification_report

```python
sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
```

| 参数          | 性质                                                        |
| ------------- | ----------------------------------------------------------- |
| y_true        | 1维数组，或标签指示器数组/稀疏矩阵，目标值。                |
| y_pred        | 1维数组，或标签指示器数组/稀疏矩阵，分类器返回的估计值。    |
| labels        | array，shape = [n_labels]，报表中包含的标签索引的可选列表。 |
| target_names  | 字符串列表，与标签匹配的可选显示名称（相同顺序）。          |
| sample_weight | 类似于shape = [n_samples]的数组，可选项，样本权重。         |
| digits        | 输出浮点值的位数。                                          |

**用法示例**

```python
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))
```

**输出结果**

```
            precision    recall  f1-score   support

    class 0       0.50      1.00      0.67         1
    class 1       0.00      0.00      0.00         1
    class 2       1.00      0.67      0.80         3

avg / total       0.70      0.60      0.61         5
```

其中列表左边的一列为分类的标签名，右边support列为每个标签的出现次数，avg / total行为各列的均值．

**参数说明**

1. **Precision**

   **精确率**是针对我们**预测结果**而言的，它表示的是预测为正的样本中**有多少是真正的正样本**。那么预测为正就有两种可能了，一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP)，也就是

   ![[公式]](https://www.zhihu.com/equation?tex=P++%3D+%5Cfrac%7BTP%7D%7BTP%2BFP%7D)

   precision 体现了模型对负样本的区分能力，precision越高，说明模型**对负样本的区分能力**越强。

   

2. **Recall**

   **召回率**是针对我们原来的**样本**而言的，它表示的是样本中的**正例有多少被预测正确了**。那也有两种可能，一种是把原来的正类预测成正类(TP)，另一种就是把原来的正类预测为负类(FN)。
   																			![[公式]](https://www.zhihu.com/equation?tex=R+%3D+%5Cfrac%7BTP%7D%7BTP%2BFN%7D)

   recall 体现了分类模型对正样本的识别能力，recall 越高，说明模型**对正样本的识别能力**越强。

   <img src="https://pic1.zhimg.com/80/d701da76199148837cfed83901cea99e_720w.jpg" alt="img" style="zoom: 67%;" />

   

3. **F1-Score**

   <img src="C:\Users\Fan\AppData\Roaming\Typora\typora-user-images\image-20200314182952150.png" alt="image-20200314182952150" style="zoom:67%;" />

   ​		F1-score是对正负样本区分能力的综合，F1-score 越高，说明分类模型越稳健。

   > 比如我们常见的雷达预警系统，我们需要对雷达信号进行分析，判断这个信号是飞行器（正样本）还是噪声 （负样本）, 很显然，我们希望系统既能准确的捕捉到飞行器信号，也可以有效地区分噪声信号。所以就要同时权衡recall 和 precision这两个指标，如果我们把所有信号都判断为飞行器，那 recall 可以达到1，但是precision将会变得很低（假设两种信号的样本数接近），可能就在 0.5 左右，那F1-score 也不会很高。

   ​	有的时候，我们对recall 与 precision 赋予不同的权重，表示对分类模型的偏好：

   ![image-20200314183641514](C:\Users\Fan\AppData\Roaming\Typora\typora-user-images\image-20200314183641514.png)

   > 可以看到，当 β=1，那么Fβ就退回到F1了，β 其实反映了模型分类能力的偏好，β>1的时候，precision的权重更大，为了提高Fβ，我们希望precision 越小，而recall 应该越大，说明模型更偏好于提升recall，意味着模型更看重对正样本的识别能力； 而 β<1的时候，recall 的权重更大，因此，我们希望recall越小，而precision越大，模型更偏好于提升precision，意味着模型更看重对负样本的区分能力。

   ​		recall越大，越不能放弃飞行器的识别，要识别所有的（宁愿把噪声识别出来），此时precision就低了。

   ​		precision越大，说明不要把噪声识别成了飞行器。只要识别是飞行器，那么就是飞行器。

   

4. **Accuracy**

   预测对的 / 所有 

> *假如某个班级有男生* **80** *人, 女生***20***人, 共计* **100** *人. 目标是找出所有女生. 现在某人挑选出* **50** *个人, 其中* **20** *人是女生, 另外还错误的把 30 个男生也当作女生挑选出来了. 作为评估者的你需要来评估(***evaluation***)下他的工作*

​		accuracy 需要得到的是此君**分正确的人**占**总人数**的比例

​		我们可以得到:他把其中70(20女+50男)人判定正确了, 而总人数是100人，所以它的 accuracy 就是70 %(70 / 100).



## preprocessing.MinMaxScaler

*class* `sklearn.preprocessing.``MinMaxScaler`(*feature_range=(0*, *1)*, *copy=True*)

| Parameters                                          | Introduction                                                 |
| --------------------------------------------------- | ------------------------------------------------------------ |
| **feature_range**: tuple (min, max), default=(0, 1) | Desired range of transformed data.                           |
| **copy**: bool, default = True                      | Set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array). |

| Attributes          | Introduction                                                 |
| ------------------- | ------------------------------------------------------------ |
| **min_**            | Per feature adjustment for minimum. Equivalent to `min - X.min(axis=0) * self.scale_` |
| **scale_**          | Per feature relative scaling of the data. Equivalent to `(max - min) / (X.max(axis=0) - X.min(axis=0))` |
| **data_min_**       | Per feature minimum seen in the data                         |
| **data_max_**       | Per feature maximum seen in the data                         |
| **data_range_**     | Per feature range `(data_max_ - data_min_)` seen in the data |
| **n_samples_seen_** | The number of samples processed by the estimator. It will be reset on new calls to fit, but increments across `partial_fit` calls. |

| Methods                                                | Introduction                                                 |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| `fit`(*self*, *X*, *y=None*)                           | Compute the minimum and maximum to be used for later scaling. |
| `fit_transform`(*self*, *X*, *y=None*, ***fit_params*) | Fit to data, then transform it.                              |
| `inverse_transform`(*self*, *X*)                       | Undo the scaling of X according to feature_range.            |
| `transform`(*self*, *X*)                               | Scale features of X according to feature_range.              |



## svm.SVC

*class* `sklearn.svm.``SVC`(*C=1.0*, *kernel='rbf'*, *degree=3*, *gamma='scale'*, *coef0=0.0*, *shrinking=True*, *probability=False*, *tol=0.001*, *cache_size=200*, *class_weight=None*, *verbose=False*, *max_iter=-1*, *decision_function_shape='ovr'*, *break_ties=False*, *random_state=None*)

**C**  惩罚系数，对误差的宽容度

- C越高，说明越不能容忍出现误差，容易出现过拟合。
- C太小，容易欠拟合

**gamma**

​	选择RBF作为核函数后，该函数自带的一个参数，隐含地决定了数据映射到新的特征空间的分布。

- 如果gamma设的**太大**（**支持向量少**），方差会很小，高斯分布“高瘦”，会造成只作用于支持向量样本附近，对未知样本的分类效果很差。存在训练准确率可以很高，但是测试准确率不高的情况，出现**过训练**。
- 如果gamma设的**太小**（**支持向量多**），则会造成平滑效应过大，无法在训练集上获得很高的准确率。



| Attributes           | Introduction                                                 |
| -------------------- | ------------------------------------------------------------ |
| **support_**         | Indices of support vectors.                                  |
| **support_vectors_** | Support vectors.                                             |
| **n_support_**       | Number of support vectors for each class.                    |
| **fit_status_**      | 0 if correctly fitted, 1 otherwise (will raise warning)      |
| **classes_**         | The classes labels.                                          |
| **class_weight_**    | Multipliers of parameter C for each class. Computed based on the `class_weight` parameter. |
| **shape_fit_**       | Array dimensions of training vector `X`.                     |

| Methods                                         | Introduction                                                 |
| ----------------------------------------------- | ------------------------------------------------------------ |
| `decision_function`(*self*, *X*)                | Evaluates the decision function for the samples in X.        |
| `fit`(*self*, *X*, *y*, *sample_weight=None*)   | Fit the SVM model according to the given training data.      |
| `get_params`(*self*, *deep=True*)               | Get parameters for this estimator.                           |
| `predict`(*self*, *X*)                          | Perform classification on samples in X.                      |
| `score`(*self*, *X*, *y*, *sample_weight=None*) | Return the **mean accuracy** on the given test data and labels. |
| `set_params`(*self*, ***params*)                | Set the parameters of this estimator.                        |

