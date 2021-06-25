"""
    Analysis the model predict with ture lable.

"""

from os import remove
import pandas as pd
import numpy as np

from evulation import RMSE_and_MAE



def analys_by_group(data_path,Key="userId"):
    """
    split users or items to group by num of rating
    Key="userId" or "movieId"
    """
    df=pd.read_csv(data_path)

    list_u=[]
    dict_len=dict()
    dict_group=dict()

    User_Group={"<32":[],
                "32-64":[],
                "64-128":[],
                "128-256":[],
                "256-512":[],
                "512-1024":[],
                ">1024":[]}

    df_src=pd.read_csv("data/ml-1m/n_ratings.csv")

    # 将每一个用户进行切分，并进行统计
    for uid,group in df.groupby([Key]):
        list_u.append(uid)
        # dict_len[u] = len(group)*5
        dict_group[uid]=group

    # 统计每个用户的长度
    for uid,group in df_src.groupby([Key]):
        dict_len[uid] = len(group)

    # 划分用户组
    for u,u_ratings in dict_len.items():
        if u_ratings<32:
            User_Group["<32"].append(u)
        elif u_ratings<64:
            User_Group["32-64"].append(u)
        elif u_ratings<128:
            User_Group["64-128"].append(u)
        elif u_ratings<256:
            User_Group["128-256"].append(u)
        elif u_ratings<512:
            User_Group["256-512"].append(u)
        elif u_ratings<1024:
            User_Group["512-1024"].append(u)
        else:
            User_Group[">1024"].append(u)

    dict_evl={
        "group_name":[],
        "count_user":[],
        "count_rating":[],
        "RMSE":[],
        "MAE":[]
    }
    # 计算用户组的评价指标
    for group_name,group in User_Group.items():
        lables=[]
        preds=[]
        if len(group)==0:
            continue
        for one in group:
            # dict_evl["list_user"].append(one)
            if one not in dict_group:
                continue
            one=dict_group[one]
            lables+=one.rating.to_list()
            preds+=one.pred.to_list()
           
        rmse,mae=RMSE_and_MAE(preds,lables)
        
        dict_evl["group_name"].append(group_name)
        dict_evl["count_user"].append(len(group))
        dict_evl["count_rating"].append(len(lables))
        dict_evl["RMSE"].append(rmse)
        dict_evl["MAE"].append(mae)
    return dict_evl


def good_and_bad(data_path,Key="userId"):
    """
    select best100 and last100 users or items by rmse and mae
    Key="userId" 或者 "movieId"
    """
    df=pd.read_csv(data_path)


    list_u=[]
    dict_len=dict()
    dict_group=dict()

    # dict_rmse=dict()
    # dict_mae=dict()
    list_rmse=[]
    list_mae=[]
    df_src=pd.read_csv("data/ml-1m/n_ratings.csv")

    # 将每一个用户进行切分，并进行统计
    for uid,group in df.groupby([Key]):
        list_u.append(uid)
        # dict_len[u] = len(group)*5
        dict_group[uid]=group
        preds=group.pred.to_list()
        lables=group.rating.to_list()
        rmse,mae=RMSE_and_MAE(preds,lables)
        list_rmse.append([uid,rmse])
        list_mae.append([uid,mae])
    # 统计每个用户的长度
    for uid,group in df_src.groupby([Key]):
        dict_len[uid] = len(group)

    sorted_list_rmse=sorted(list_rmse,key=lambda x:x[1])

    sorted_list_mae=sorted(list_mae,key=lambda x:x[1])

    top100_rmse=sorted_list_rmse[:100]
    last100_rmse=sorted_list_rmse[-100:-1]

    top100_mae=sorted_list_mae[:100]
    last100_mae=sorted_list_mae[-100:-1]

    res=dict()

    def show(x,y):
        count=0
        v=np.mean([i[1] for i in y ])
        for one in y:
            count+=dict_len[one[0]]
        # print("{}:{}\trating count:{}".format(x,v,count))
        res[x]=[v,count]
    show("top100RMSE",top100_rmse)
    show("last100RMSE",last100_rmse)
    show("top100MAE",top100_mae)
    show("last100MAE",last100_mae)
    return res
   

def mean_good_and_bad(path,key):


    df=pd.DataFrame()
    res=[]
    for i in range(1,6):
        print("analysing {}...".format(i))
        res.append(good_and_bad(path.format(i),Key=key))
    one=res[0]
    for key in one:
        v=[]
        c=[]
        for racd in res:
            v.append(racd[key][0])
            c.append(racd[key][1])
        print("{}:{}\tcount:{}".format(key,np.mean(v),np.mean(c)))


def mean_analys_by_group(path,key):
    
    df=pd.DataFrame()
    for i in range(1,6):
        print("analysing {}...".format(i))
        res=analys_by_user_group(path.format(i),Key=key)
        df=pd.concat([df,pd.DataFrame(res)])
    
    for name,group in df.groupby(["group_name"]):
        rmse=group.RMSE.tolist()
        mae=group.MAE.tolist()
        print("{}\tRMSE:{}\tMAE:{}".format(name,np.mean(rmse),np.mean(mae)))
