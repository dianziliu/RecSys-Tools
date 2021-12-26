from _typeshed import ReadOnlyBuffer
from numpy.lib.ufunclike import isneginf
import pandas as pd
import numpy as np
import random

from pandas.core import groupby
from pandas.core.algorithms import mode


def all_sampler(n_items,train_path,test_path):
    # 全集
    res=dict()
    train_data=pd.read_csv(train_path)
    test_data=pd.read_csv(test_path)
    all_items=set(range(n_items))
    data=pd.concat([train_data,test_data])
    for uid,group in data.groupby(["userId"]):
        his=group.movieId.unique().tolist()
        ngs=list(
            all_items-set(his)
        )
        res[uid]=ngs
    return res

def k_sampler(n_items,train_path,test_path,ratio):
    # 按照比例进行采样
    res=dict()
    train_data=pd.read_csv(train_path)
    test_data=pd.read_csv(test_path)
    all_items=set(range(n_items))
    data=pd.concat([train_data,test_data])
    for uid,group in data.groupby(["userId"]):
        his=group.movieId.unique().tolist()
        ngs=list(
            all_items-set(his)
        )
        res[uid]=random.choices(ngs,k=ratio*len(his))
    return res


def sampler(n_items,train_path,test_path,mode="all"):
    if mode=="all":
        return all_sampler(n_items,train_path,test_path)
    elif isinstance(mode,int):
        return k_sampler(n_items,train_path,test_path,mode)
    else:
        assert "not support!"
