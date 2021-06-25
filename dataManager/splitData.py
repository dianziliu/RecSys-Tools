"""
version：0.0.1
author：刘俊锐
功能：提供了多种工具用于划分数据集
"""

from itertools import count
from random import choices,choice,randint

import joblib
from numpy.lib import index_tricks
from numpy.lib.arraysetops import isin
import pandas as pd
from sklearn.model_selection import train_test_split

# 就不在这里支持多线程了。可以提供一个工具
from multiprocessing import Pool,cpu_count


def splitData(mode,data_path,epoch=1,dir="",**kwargs):
    """一个总的接口，用于划分数据集
    mode: 用于指定划分方式
    data_path：指定数据地址
    epoch：划分次数
        """
    methods={
        "sample":splitData_sample
    }
    df=pd.read_csv(data_path)
    if epoch==1:
        return methods[mode](kwargs)

    # 多线程处理数据
    
    # 防止占满后卡死
    p=Pool(min(epoch,cpu_count()-2))

    for i in range(epoch):
        p.apply_async(methods[mode](kwargs))
    p.close()
    p.join()
    
    

def splitData_sample(df,test_size,dump=False,train_path="",test_path="",key=None):
    """
        简单的划分数据集，不做任何额外处理
        """
    train_size=1-test_size
    if not key:
        train_data=df.sample(frac=train_size)
        test_data=df[~df.index.isin(train_data.index)]
    else:
        train_data=[]
        test_data=[]
        for id,group in df.groupby([key]):
            this_train_data=group.sample(frac=train_size)
            train_data.append(this_train_data)
            test_size.append(group[~group.index.isin(this_train_data.index)])
        train_data=pd.concat(train_data)
        test_data=pd.concat(test_data)
    if dump:
        train_data.to_csv(train_path,index=False)
        test_data.to_csv(test_path,index=False)
    return train_data,test_data




def process_data_by_numN(path,sep,n_items,N,train_path,test_path,ngs_path,n_ng=4):

    df=pd.read_csv(path,sep=sep)

    train_file=open(train_path,"w")
    test_file=open(test_path,"w")
    test_ng=ngs_path
    d=dict()

    df2=pd.DataFrame()

    iids=set([i for i in range(n_items)])

    for uid,group in df.groupby(["userId"]):
        h=list(group.movieId.unique())
        l=len(h)
        if l<=N:
            continue
        # 采集负样本范围
        ng=list(set(iids)-set(h))
        ngs=choices(ng,k=l*5)
        # 划分训练集测试集
        g=[i for i in range(l)]
        test_index=choices(g,k=N)
        test=group.ix[test_index]
        for one in test_index:
            del h[one]
        train=h
        # 写入训练集
        pgs=choices(train,k=l*5)
        for i in range(len(train)):
            for j in range(n_ng): #here control num of negative sample
                train_file.write("{},{},{},{}\n".format(
                    uid, train[i], pgs[i*n_ng+j], ngs[i*n_ng+j]))
        df2=pd.concat([df2,test]) 

    train_file.close()
    df2.to_csv(test_file,index=False)
    joblib.dump(d,test_ng)

  
def process_data_with_ngs(path,sep,n_items,train_path,test_path,ngs_path,test_size=0.2,n_ng=4):

    df=pd.read_csv(path,sep=sep)

    train_file=open(train_path,"w")
    test_file=open(test_path,"w")
    test_ng=ngs_path
    d=dict()

    df2=pd.DataFrame()
    iids=[i for i in range(n_items)]

    for uid,group in df.groupby(["userId"]):
        h=list(group.movieId.unique())
        l=len(h)
        ng=list(set(iids)-set(h))
        ngs=choices(ng,k=l*5)
        
        train,test=train_test_split(group,test_size=test_size)
        train=list(train.movieId.unique())
        
        pgs=choices(train,k=l*5)
        for i in range(len(train)):
            for j in range(n_ng): #here control num of negative sample
                train_file.write("{},{},{},{}\n".format(
                    uid, train[i], pgs[i*n_ng+j], ngs[i*n_ng+j]))
        
        ngs=choices(ng,k=100) #here control num of negative sample
        d[uid]=ngs
        df2=pd.concat([df2,test])
            

    train_file.close()
    df2.to_csv(test_file,index=False)
    joblib.dump(d,test_ng)

def process_data_by_split(path,sep,n_items,train_path,test_path,test_size=0.5,n_ng=4):
    
    """
    训练文件以四元组的形式保存 "u,i,postive,negative"
    测试集以原始文件格式保存.csv
    """
    df=pd.read_csv(path,sep=sep,engine="python")

    train_file=open(train_path,"w")
    # test_file=open("test_file.csv","w")
    
    df2=pd.DataFrame()

    iids=[i for i in range(n_items)]
    for uid,group in df.groupby(["userId"]):
        
        h=list(group.movieId.unique())
        l=len(h)
        ng=list(set(iids)-set(h))
        ngs=choices(ng,k=l*5)
        
        train,test=train_test_split(group,test_size=test_size)
        train=list(train.movieId.unique())
        pgs=choices(train,k=l*5)
        for i in range(len(train)):
            for j in range(n_ng):
                train_file.write("{},{},{},{}\n".format(
                    uid, train[i], pgs[i*n_ng+j], ngs[i*n_ng+j]))
        df2=pd.concat([df2,test])

    train_file.close()
    df2.to_csv(test_path,index=False)
