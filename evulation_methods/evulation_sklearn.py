"""
    writhing by sk-learn
"""
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import ndcg_score
from sklearn.metrics import precision_score,recall_score,f1_score
import pandas as pd

def MAE(u_tures,u_preds):
    y_tures=[]
    [y_tures.extend(i) for i in u_tures]
    y_preds=[]
    [y_preds.extend(i) for i in u_preds]
    return mean_absolute_error(y_true,y_pred)

def RMSE(u_tures,u_preds):
    """
        使用sklean的方法计算rmse
    """
    y_tures=[]
    [y_tures.extend(i) for i in u_tures]
    y_preds=[]
    [y_preds.extend(i) for i in u_preds]
    return sqrt(mean_squared_error(y_ture,y_pred))

def Precision(u_tures,u_preds):
    pass

def Recall(u_tures,u_preds):
    pass

def NDCG(y_rating,y_scores):
    """
    y_rating：2-D array/list, 其中元素的值标识该位置的物品的分数
    y_scores:2-D scores，其中元素的值标识该位置的物品的预测值
    """

    return ndcg_score(y_rating,y_scores)



def evulation_without_ng(test_file,model,need_method):

    df = pd.read_csv(test_file)
    
    u_tures=[]
    u_preds=[]
    for uid, group in df.groupby(["userId"]):
        ratings=[]
        scores = []
        for row,one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            ratings.append(r)
            scores.append(model.predict(u,i))
            u_tures.append(ratings)
            u_preds.append(scores)
    Methods={
        "RMSE":RMSE,
        "MAE":MAE,
        "NDCG":NDCG
    }
    Results=dict()
    for method in need_method:
        if method in Methods:
            Results[method]=Methods[method](u_tures,u_preds)
        else:
            print("{} not exist!".format(method))
    return Results

def evulation_with_ng(test_file,model,need_method):

    df = pd.read_csv(test_file)
    
    u_tures=[]
    u_preds=[]
    for uid, group in df.groupby(["userId"]):
        ratings=[]
        scores = []
        for row,one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            ratings.append(r)
            scores.append(model.predict(u,i))
            u_tures.append(ratings)
            u_preds.append(scores)
    Methods={
        "RMSE":RMSE,
        "MAE":MAE,
        "NDCG":NDCG
    }
    Results=dict()
    for method in need_method:
        if method in Methods:
            Results[method]=Methods[method](u_tures,u_preds)
        else:
            print("{} not exist!".format(method))
    return Results



if __name__ =="__main__":
    s1=[[1,2,3,2]]
    s2=[[.1,.3,.2,.3],
        [.3,.1,.3,.1]]

    print(ndcg_score(s1,s2))