from math import log2
from multiprocessing.pool import Pool
from random import shuffle

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_pandas


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

def MAP_evulation_Alls(model, Ks:list, test_path,n_itmes):
    aps=dict()
    for K in Ks:
        aps[K]=[]
    df = pd.read_csv(test_path)

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
        for K in Ks:
            aps[K].append(AP(K,scores,ids))
    for K in Ks:
        print("model{0} mAP@{1}:{2}".format(model.name, K, np.mean(aps[K])))



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


def Precision_Recall(K,scores,right):
    s=sorted(scores,key=lambda x: x[1],reverse=True)
    s=[i[0] for i in s[:K]]
    m=set(s) & set(right)
    p=len(m)/K
    r=len(m)/len(right)
    return p,r


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



def pr_ndcg_evulation(model, K, test_path, test_ng):
    precisions = []
    recalls = []
    ndcgs = []
    df = pd.read_csv(test_path)
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
