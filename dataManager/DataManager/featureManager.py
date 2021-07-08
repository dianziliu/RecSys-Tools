import sys
sys.path.append("DataManager")
import pickle
from sklearn.preprocessing import LabelEncoder
from encoderManager import EncoderManager,EncoderManagerPlus
from featureConfig.feature1 import *

import os
class FeatureManager(EncoderManager):
    name="FeatureManager"

class FeatureManagerPlus(EncoderManagerPlus):
    name="FeatureManager"



# class BigManagerFeature():
#     """
#         改用dict实现
#         增加了历史行为记录功能
#         """
#     name="Manager"
#     defaultPath=name+".pkl"
#     def __init__(self,df,features,lbePath="",dir="") -> None:
#         self.defaultPath=os.path.join(dir,self.name+".pkl")
#         # 特征管理器
#         self.features_managers=dict()
#     def init(self,df,features):
#         # 第一个特征为用户特征
#         uids=df[features[0]].unique().tolist()
#         self.features_managers[features[0]]=UserManager(uids)
#         for feature in features[1:]:
#             ids=df[feature].unqiue().tolist()
#             self.features_managers[feature]=FeatureManagerPlus(ids)

#     def reset(self,ids):
#         for one in ids:
#             if one in self.word2idx:
#                 continue
#             self.word2idx[one]=self.max_idx
#             # 初始化记录表
#             self.history[self.max_idx]=[]
#             self.idx2word.append(one)
#             self.max_idx+=1
            
#     def get_words(self):
#         return list(self.word2idx.keys())
#     def transform(self,id):
#         # 获取转换后的Id
#         if isinstance(id,list):
#             if isinstance(id[0],int):
#                 return id
#             return [self.word2idx[one] for one in id]
#         else:
#             if isinstance(id,int):
#                 return id
#             return self.word2idx[id]
    
#     def re_transform(self,new_id):
#         # 返回原始编码
#         if isinstance(new_id,list):
#             return [self.idx2word[one] for one in new_id]
#         else:
#             return self.idx2word[new_id]
#     def reload(self,path=""):
#         if path=="":
#             path=self.defaultPath
#         with open(path,"rb") as f:
#             self.lbe=pickle.load(f)
#     def store(self,path=""):
#         if path=="":
#             path=self.defaultPath
#         with open(path,"wb") as f:
#             pickle.dump(self.lbe,f)
#     def get_params(self):
#         return self.word2idx,self.idx2word
    
#     def get_history(self,user):
#         if not isinstance(user,int):
#             user=self.transform(user)
#         return self.history[user]
#     def record_history(self,dataset,feature1,feature2):
#         """
#         要求dataset为pd.DataFrame格式
#         并且，是对uid和iid转换之后的数据
#         """
#         for user,group in dataset.groupby([feature1]):
#             items=group[feature2].tolist()
#             self.history[user].extend(items)




