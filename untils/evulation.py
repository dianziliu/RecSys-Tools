"""
    Some evulations of Recommender Systems
    XXX_evualtion 为封装好的接口，用于评价指标计算
    XXX_s 加入了对多K值的处理
    TODO：使用多线程进行加速计算
"""



from collections import defaultdict
from math import log2
from random import shuffle

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .sampler import sampler



def AP(K,scores, ids):
    """
    scores:[iId,score,rating]
    ids:用户的测试集
    """
    s=sorted(scores,key=lambda x: x[1],reverse=True)
    s=[i[0] for i in s[:K]] 
    sum=0
    hits=0
    for i in range(K):
        if s[i] in ids:
            hits+=1
            sum+=hits/(i+1)
    if hits==0:
        return 0
    else:
        return sum/hits


def APs(Ks, scores_s, ids_s):
    aps = dict()
    for k in Ks:
        aps[k] = []
        for scores, ids in zip(scores_s, ids_s):
            aps[k].append(AP(k, scores, ids))
    return aps


def DCG(K, l, w,method=0):
    s = sorted(l, key=lambda x: x[1], reverse=True)

    if method:
        x=[pow(i[2],2)-1 for i in s[:K]]
    else:
        x=[i[2] for i in s[:K]]
    return np.dot(x, w[:K])

def iDCG(K, l, w,method=0):
    s = sorted(l, key=lambda x: x[2], reverse=True)
    if method:
        x=[pow(i[2],2)-1 for i in s[:K]]
    else:
        x=[i[2] for i in s[:K]]
    return np.dot(x, w)

def nDCG(K, l,method=0):
    """
    k:int
    l:list[(int ,int ,int)]
    """
    if len(l)<K:
        K=len(l)
    w = [1/log2(2+i) for i in range(K)]
    dcg = DCG(K, l, w,method)
    idcg = iDCG(K, l, w,method)
    return dcg/idcg

def nDCGs(Ks,ls,method=0):
    res=dict()
    for k in Ks:
        res[k]=[]
        for l in ls:
            res[k].append(nDCG(k,l,method))
    return res




def Precision_Recall(K,scores,right):
    s=sorted(scores,key=lambda x: x[1],reverse=True)
    s=[i[0] for i in s[:K]]
    m=set(s) & set(right)
    p=len(m)/K
    r=len(m)/len(right)
    return p,r

def Precision_Recalls(Ks,scores_s,right_s):
    res=dict()
    for k in Ks:
        res[k]=[]
        for scores,right in zip(scores_s,right_s):
            res[k].append(Precision_Recall(k,scores,right))
    return res

