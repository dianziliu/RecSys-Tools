# 工具集合
## 提供了标准的评论工具
## 提供了一些其他工具

# 评价指标说明

本文的目的是对目前推荐系统以及机器学习中的一些评价指标进行讲解和说明。谓之前人栽树，后人纳凉。

## 分类指标

### 精度、召回、F1、Acc、AUC、ROC

#### 二分类

> 基本说明：在二分类中，样本的标签分为正类和负类两类。同时，模型的预测值也分为正负两类。但是，模型的预测值可能和样本的标签是不一致的，因此产生了评价问题。

> 为了方便说明，将预测值与真实值之间的关系进行如下规定：
>
> 1. TP：表示预测为正，且标签为正
> 2. TF：表示预测为负，且标签为负
> 3. FP：表示预测为正，但标签为负
> 4. FF：表示预测为负，但标签为正
>    以上四种情况涵盖了一个二分类模型中真实值与预测值之间的所有关系的可能性。

精度:

$$
P=\frac{TP}{TP+FP}

$$

召回：

$$
R=\frac{TP}{TP+FF}

$$

F1：

$$
F_1=\frac{2*P*R}{P+R}

$$

$F_{\beta}$:

$$
F_{\beta}=\frac{(1+\beta^2)*P*R}{\beta^2*P*R}

$$

ROC：在二分类模型中，模型的预测值往往是一个连续的实数值，一般选择一个阈值将其二分化。因此，阈值的影响将会显著影响PR等指标的计算。

> 1. ROC是一个曲线的面积。
> 2. 特异性：负样本被预测为负样本的比例
>    $$
>    Sp=\frac{TF}{TF+FP}
>
>    $$
> 3. 横轴为：1-特异性
> 4. 纵轴为：精度
> 5. 特性：该曲线过（0，0）点和（1，1）点，随机方法是一条直线。


AUC：直观的反应了ROC曲线表达的分类能力，是ROC曲线的面积。

ACC：训练精度和测试精度的加权结果。


#### 多分类

问题：当二分类扩展为多分类时，对评价指标的计算将出现一些不同。不同的模型从不同的角度优化多分类模型，导致了不同的处理方式。

1. 将预测值进行二分判断，即将原始标签视为正，其余视为负，这样可以直接复用二分类的结果。
2. 针对每一类，进行判断，分别求评价指标并进行平均。

#### 多标签

### 标准接口

sklearn: metrics

P: [https://scikit-learn.org/dev/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score%E2%80%B8)

R: [https://scikit-learn.org/dev/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score%E2%80%B8)

F1: [https://scikit-learn.org/dev/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score%E2%80%B8)

ACC: [https://scikit-learn.org/dev/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score%E2%80%B8)

ROC_ACC: [https://scikit-learn.org/dev/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score%E2%80%B8)


tensorflow/keras


### 在RS中的应用方式

在RS中使用分类指标分为两种情况：基于排序的视角和基于分类的视角。

#### 基于排序的视角

推荐系统一般面向用户，并选取Top-K个结果作为最终结果。因此，评价结果产生了了一定的变化，RS中不在需要阈值，而是以排序结果进行选择。

因此，P->P@K, R->R@K，...


#### 基于分类的视角

## 排序指标

基于Top-K的RS方法将推荐问题建模为排序问题。


### Hit、 mAP、 nDCG

hit：

$$
hit@K=\frac{1}{|\mathcal{U}|}\sum_{u \in \mathcal{U}}{I(pred@K,list_u)}

$$

mAP：

$$
mAP=\frac{1}{|\mathcal{U}|}\sum_{u \in \mathcal{U}}{P(pred@K,list_u)}

$$

nDCG：



### 标准接口


ndcg:

[https://scikit-learn.org/dev/modules/generated/sklearn.metrics.ndcg_score.html](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.ndcg_score.html)

```
pip install --user --upgrade tensorflow_ranking
```

官方文档：[https://github.com/tensorflow/ranking](https://github.com/tensorflow/ranking%E2%80%B8)

## 回归指标

### RMSE、 MAE

### 标准接口
