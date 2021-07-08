"""
定义了一些工具
"""
from featureConfig.feature1 import *
import pandas as pd
import numpy

def transFeature(data,feature,feature_manager):
    # 将原始数据转化为编码后的数据
    users=data[feature].tolist()
    users=feature_manager.transform(users)
    data[feature]=users


def transData2Frame(data,user_manager,item_manager):
    # 将原始数据转化为编码后的数据
    transFeature(data,user_features[0],user_manager)
    transFeature(data,item_features[0],item_manager)

    # users=data[user_features[0]].tolist()
    # users=user_manager.transform(users)
    # data[user_features[0]]=users
    # # data[user_features[0]].repalce(users)
    # items=data[item_features[0]].tolist()
    # items=item_manager.transform(items)
    # data[item_features[0]]=items
    # # data[item_features[0]].replace(items)

if __name__=="__main__":
    pass
