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
from numpy.lib.function_base import sort_complex
import pandas as pd
from pandas.core.groupby.generic import ScalarResult
from tqdm import tqdm

def evulation(model,n_items,test_path,evulation_mode:dict)->dict:
    """
        总的封装函数，通过传入训练好的模型，测试集和需要计算的评价指标，得到dict类型的返回结果。
        model：
            1. 要求支持 predict方法。（最好支持predicts方法，批量进行预测）
            2. 要求支持 name属性
        test_path：
            1. 要求以csv文件格式进行存储，用userId，movieId，rating字段进标识。
    """
    evulation_method={
        "ndcg":nDCGs,
        "aps":APs,
    }
    df = pd.read_csv(test_path)
    # 结果暂存
    scores_s=[]
    ids_s=[]

    # 计算预测值
    for uid, group in df.groupby(["userId"]):
        scores = []
        # TODO：这里可以使用predicts方法进行加速
        for i in range(n_items):
            scores.append([i, model.predict(uid, i), 0])
        # ids 为用户历史行为
        ids = group.movieId.unique()
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
        shuffle(scores)
        scores_s.append(scores)
        ids_s.append(ids)
    res=dict()
    # 计算每一个指标
    for k,v in evulation_mode.items():
        res[k]=evulation_method[k](v,scores_s,ids_s)
    pass


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

def MAP_evulation_Alls(model, Ks:list, test_path,n_itmes,show=True):
    """

    """
 
    df = pd.read_csv(test_path)
    # 结果暂存
    scores_s=[]
    ids_s=[]

    # 计算预测值
    for uid, group in df.groupby(["userId"]):
        scores = []
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        # ids 为用户历史行为
        ids = group.movieId.unique()
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
        shuffle(scores)
        scores_s.append(scores)
        ids_s.append(ids)
    # 
    aps=APs(Ks,scores_s,ids_s)

    # 打印结果
    if show:
        for K in Ks:
            print("model{0} mAP@{1}:{2}".format(model.name, K, np.mean(aps[K])))
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

def ndcg_evaluations(model, Ks, test_file,method=0,show=True):
    # 将Ks统一为list进行处理
    if not isinstance(Ks,list):
        Ks=[Ks]
    if not isinstance(Ks[0],int):
        assert "Ks value must be 'int' type"
    df = pd.read_csv(test_file)
    scores_s=[]
    for uid, group in df.groupby(["userId"]):
        scores = []
        for row,one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores.append((i, model.predict(u, i), r))
        # ndcgs.append(nDCG(K, scores,method))
        scores_s.append(scores)
    ndcgs=nDCGs(Ks,scores_s,method)
    if show:
        for k in Ks:
            print("model{} nDCG@{}:{}".format(model.name, k, np.mean(ndcgs)))



def ndcg_evaluation(model, K, test_file,method=0):
    df = pd.read_csv(test_file)
    ndcgs = []
    for uid, group in df.groupby(["userId"]):
        scores = []
        for row,one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores.append((i, model.predict(u, i), r))
        ndcgs.append(nDCG(K, scores,method))
    
    print("model{} nDCG@{}:{}".format(model.name, K, np.mean(ndcgs)))



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

def pr_evulation(model, K, test_path, test_ng):
    precisions = []
    recalls = []
    df = pd.read_csv(test_path)
    ngs = joblib.load(test_ng)
    for uid, group in df.groupby(["userId"]):
        scores = []
        ids = group.movieId.unique()
        for i in ids:
            scores.append((i, model.predict(uid, i)))
        ng = ngs[uid]
        for j in ng:
            scores.append((i, model.predict(uid, j), 0))
        shuffle(scores)
        p, r = Precision_Recall(K, scores, ids)
        precisions.append(p)
        recalls.append(r)

    print("model{} Precision@{}:{},Recall@{}:{}".format(
        model.name, K, np.mean(precisions), K, np.mean(recalls)))




def pr_ndcg_evulation(model, K, test_path, test_ng):
    # 
    precisions = []
    recalls    = []
    ndcgs      = []
    #
    df  = pd.read_csv(test_path)
    ngs = joblib.load(test_ng)
    for uid, group in df.groupby(["userId"]):
        scores = []
        ids = group.movieId.unique()
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores.append((i, model.predict(u, i), r))
        ng = ngs[uid]
        for j in ng:
            scores.append((i, model.predict(uid, j), 0))
        shuffle(scores)
        p, r = Precision_Recall(K, scores, ids)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(nDCG(K, scores))
    print("model{0} Precision@{1}:{2},Recall@{1}:{3},nDCG@{1}:{4}".format(
        model.name, K, np.mean(precisions), np.mean(recalls), np.mean(ndcgs)))


def pr_ndcg_evulation_All(model, K, test_path, n_itmes):
    precisions = []
    recalls = []
    ndcgs = []
    ndcgs2=[]
    df = pd.read_csv(test_path)
    # All_items=[i for i in range(n_itmes)]

    for uid, group in df.groupby(["userId"]):
        scores = []
        scores2=[]
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        ids = group.movieId.unique()
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
            scores2.append(scores[i])
        shuffle(scores)
        p, r = Precision_Recall(K, scores, ids)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(nDCG(K, scores))
        ndcgs2.append(nDCG(K, scores2))
    print("model{0} Precision@{1}:{2},Recall@{1}:{3},nDCG@{1}:{4},nDCG2@{1}:{5}".format(
        model.name, K, np.mean(precisions), np.mean(recalls), np.mean(ndcgs),np.mean(ndcgs2)))

def pr_ndcg_evulation_Alls(model, Ks, test_path, n_itmes):
    precisions = {}
    recalls = {}
    ndcgs = {}
    ndcgs2={}
    for one in Ks:
        precisions[one] = []
        recalls[one] = []
        ndcgs[one] = []
        ndcgs2[one]=[]
    df = pd.read_csv(test_path)
    # All_items=[i for i in range(n_itmes)]

    for uid, group in df.groupby(["userId"]):
        scores = []
        scores2=[]
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        ids = group.movieId.unique()
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
            scores2.append(scores[i])
        shuffle(scores)
        for K in Ks:
            p, r = Precision_Recall(K, scores, ids)
            precisions[K].append(p)
            recalls[K].append(r)
            ndcgs[K].append(nDCG(K, scores))
            ndcgs2[K].append(nDCG(K, scores2))
    for K in Ks:
        print("model{0} Precision@{1}:{2},Recall@{1}:{3},nDCG@{1}:{4},nDCG2@{1}:{5}".format(
            model.name, K, np.mean(precisions[K]), np.mean(recalls[K]), np.mean(ndcgs[K]),np.mean(ndcgs2[K])))


def ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_itmes):

    aps=dict()
    ndcgs=dict()
    
    for K in Ks1:
        ndcgs[K]=[]
    for K in Ks2:
        aps[K]=[]
    df = pd.read_csv(test_path,engine="python")
    history=dict()
    # 排除训练集的影响
    df2=pd.read_csv(train_path,names=["userId","moviei","moviej","moviek"],engine="python")
    for uid, group in df2.groupby(["userId"]):
        ids=group.moviei.unique()
        history[uid]=ids
    for uid, group in tqdm(df.groupby(["userId"]),ncols=50):
        ids = group.movieId.unique()
        # mAP的检测综合了所以样本，包括未观测的
        scores = []
        # 用于ndcg，只考虑测试集中的数据
        scores2=[]
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        # 排除训练集的影响
        for i in history[uid]:
            scores[i][1]=0
        
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
            scores2.append(scores[i])
        shuffle(scores)
        for K in Ks1:
            ndcgs[K].append(nDCG(K,scores2))
        for K in Ks2:
            aps[K].append(AP(K,scores,ids))
            
    for K in Ks1:
        print("model{0} NDCG@{1}:{2}".format(model.name, K, np.mean(ndcgs[K])))
    for K in Ks2:
        print("model{0} mAP@{1}:{2}".format(model.name, K, np.mean(aps[K])))
    return ndcgs,aps


def ndcg_map_evulations_(model,Ks1,Ks2,train_path,test_path,n_itmes):
    aps=dict()
    ndcgs=dict()
  
    for K in Ks1:
        ndcgs[K]=[]
    for K in Ks2:
        aps[K]=[]
    df = pd.read_csv(test_path,engine="python")
    history=dict()
    df2=pd.read_csv(train_path,names=["userId","moviei","moviej","moviek"],engine="python")
    for uid, group in df2.groupby(["userId"]):
        ids=group.moviei.unique()
        history[uid]=ids
    for uid, group in tqdm(df.groupby(["userId"]),ncols=50):
        ids = group.movieId.unique()
        scores = []
        scores2=[]
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        # 排除训练集的影响
        for i in history[uid]:
            scores[i][1]=0
        
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
            scores2.append(scores[i])
        shuffle(scores)
        for K in Ks1:
            ndcgs[K].append(nDCG(K,scores2))
        for K in Ks2:
            aps[K].append(AP(K,scores,ids))
            
    for K in Ks1:
        print("model{0} NDCG@{1}:{2}".format(model.name, K, np.mean(ndcgs[K])))
    for K in Ks2:
        print("model{0} mAP@{1}:{2}".format(model.name, K, np.mean(aps[K])))
    return ndcgs,aps
