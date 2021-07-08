import sys

from pandas.io.parquet import FastParquetImpl

sys.path.append("DataManager")
import os
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from encoderManager import EncoderManager, EncoderManagerPlus
from featureManager import FeatureManagerPlus
from itemManager import ItemManagerPlus
from sampler import unine_sampler
from textManager import TextManagerPlus
# from featureConfig.feature1 import *
from userManager import UserManagerPlus

from multiprocessing import Pool,cpu_count
from concurrent.futures import ProcessPoolExecutor

from utility.decorator import logger
    

class DataManager():
    """
        从pandas.DataFrame文件中建立合适的特征管理工具
        """
    name="Manager"
    defaultPath=name+".pkl"
    def __init__(self,df,features:list,user_features=["distinct_id"],item_features=["post_ID"],text_feature=[],dir="") -> None:
        self.defaultPath=os.path.join(dir,self.name+".pkl")
        # 特征管理器
        # 第一个特征为用户id
        # 第二个特征为物品id
        # 还是要将特征区分为用户特征和物品特征
        assert features[0]==user_features[0],"features[0] not user id"
        assert features[1]==item_features[0],"features[1] not item id"
        self.old_data = df
        self.features = features
        self.user_features=user_features
        self.item_features=item_features
        self.features_managers = dict()

        self.init(df, features)
        # 返回整理后的数据
        self._rebuild_data(df.copy(), features)
        if len(text_feature)>0:
            self.init_text(text_feature)
        # 过滤掉无用特征
        self.new_data=self.new_data[self.features]
        # 
        self.build_user_data()
        self.build_item_data()

    @logger("start to init features managers...",["features"],"end of init!")
    def init(self,df,features):
        """ 对数据进行编码
            """
        # 第一个特征为用户特征
        # 第二个特征为物品特征
        uids=df[features[0]].unique().tolist()
        iids=df[features[1]].unique().tolist()
        self.features_managers[features[0]]=UserManagerPlus(uids)
        self.features_managers[features[1]]=ItemManagerPlus(iids)
        for feature in features[2:]:
            ids=df[feature].unique().tolist()
            self.features_managers[feature]=FeatureManagerPlus(ids)

    @logger("start to init text managers...",["text_features"],"end of init!")
    def init_text(self,text_features):
        # 对文本数据进行处理
        self.features.append('text')
        self.features_managers['text']=TextManagerPlus()
        self.features_managers['text'].build_vocab(self.new_data,text_features)
        self.new_data.drop(columns=text_features[1:],inplace=True)
        self.new_data['text']=self.new_data[self.features[1]].astype(int)
    def _rebuild_data(self,df,features):
        # 重建数据
        for feature in features:
            manager=self.features_managers[feature]
            ids=df[feature].tolist()
            df[feature]=manager.transform(ids)
        self.new_data=df
        return df

    def rebuild_one_negative_sample(self,all_items,uid,history,ratio):
        """ 被弃用"""
        ng=unine_sampler(all_items,history,ratio)
        u_data=pd.concat([self.get_user_data(uid)] *len(ng),ignore_index=True)
        ng_data=pd.concat([self.get_item_data(iid) for iid in ng],ignore_index=True)
        # 合并数据
        ng_df=pd.concat([u_data,ng_data],axis=1)
        ng_df["label"]=0
        return ng_df

    @logger("start to rebuild data with nagative sample")
    def rebuild_data_negative_sample_pool(self,ratio=1,save_path=None):
        # 使用进程池实现 加速效果并不理想

        all_items=range(1,self.features_managers[self.features[1]].max_idx)
        new_data_ng=self.new_data.copy()
        iid_lable=self.features[1]
        new_data_ng["label"]=1

        with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
            uids=[]
            iids=[]
            for uid,group in tqdm(self.new_data.groupby([self.features[0]]),ncols=100):
                uids.append(uid)
                iids.append(group[iid_lable].unique().tolist())
                # sampled_negative_items = executor.map(
                #     self.rebuild_one_negative_sample, self, all_items, uid, group[iid_lable].unique(
                #     ).tolist(), ratio
                # )
            l=len(uids)
            res = executor.map(
                self.rebuild_one_negative_sample,[all_items]*l, uids,iids, [ratio]*l
            ,chunksize=100)            
        sampled_negative_items=[i for i in res]
        sampled_negative_items.append(new_data_ng)
        new_data_ng=pd.concat(sampled_negative_items)
        new_data_ng=new_data_ng.sort_values(by=[self.features[0]])
        if save_path:
            new_data_ng.to_csv(save_path,index=False)
        return new_data_ng


    def rebuild_data_negative_sample(self,ratio=1):
        all_items=range(1,self.features_managers[self.features[1]].max_idx)
        new_data_ng=self.new_data.copy()
        iid_lable=self.features[1]
        new_data_ng["label"]=1
        for uid,group in tqdm(self.new_data.groupby([self.features[0]]),ncols=100):
            ng=unine_sampler(all_items,group[iid_lable].unique().tolist(),ratio)
            u_data=pd.concat([self.get_user_data(uid)] *len(ng),ignore_index=True)
            ng_data=pd.concat([self.get_item_data(iid) for iid in ng],ignore_index=True)
            # 合并数据
            ng_df=pd.concat([u_data,ng_data],axis=1)
            ng_df["label"]=0
            new_data_ng=pd.concat([new_data_ng,ng_df])
        path="rsData/test4_ng.csv"
        new_data_ng.to_csv(path,index=False)
        return new_data_ng


    def get_words(self,feature):
        """
            获取某个完全索引
            """
        manager=self.features_managers[feature]
        return list(manager.word2idx.keys())

    def transform(self,feature,id):
        manager=self.features[feature]
        # 获取转换后的Id
        if isinstance(id,list):
            if isinstance(id[0],int):
                return id
            return [manager.word2idx[one] for one in id]
        else:
            if isinstance(id,int):
                return id
            return manager.word2idx[id]
    
    def get_sub_data(self,sub_features,mode="df"):
        # 获取子集
        if mode.lower()=="df":
            return self.new_data[sub_features]
        elif mode.lower() in ["tuple","list"]:
            sub_data=[]
            for feature in sub_features:
                sub_data.append(self.new_data[feature].tolist())
            return [i for i in zip(*sub_data)]

    def re_transform(self,feature,new_id):
        manager=self.features_managers[feature]
        # 返回原始编码
        if isinstance(new_id,list):
            return [manager.idx2word[one] for one in new_id]
        else:
            return manager.idx2word[new_id]
    def reload(self,path=""):
        if path=="":
            path=self.defaultPath
        with open(path,"rb") as f:
            self.lbe=pickle.load(f)
    def store(self,path=""):
        if path=="":
            path=self.defaultPath
        with open(path,"wb") as f:
            pickle.dump(self.lbe,f)
    def build_user_data(self):
        # 构建用户数据
        self.user_data=self.new_data[self.user_features].drop_duplicates(self.features[0],"first")
    def build_item_data(self):
        # 构建物品数据
        self.item_data=self.new_data[self.item_features].drop_duplicates(self.features[1],"first")
    def get_user_data(self,uid):
        return self.user_data[self.user_data[self.features[0]]==uid]
    
    def get_item_data(self,iid):
        # 获取物品的相关数据        
        return self.item_data[self.item_data[self.features[1]]==iid]

    def rubuild(df):
        """TODO: 将训练集和测试集进行翻译
        1. 可以将前一半视为训练集，后一半视为测试集
        """
        pass

if __name__=="__main__":

    
    features = ["distinct_id",
                "post_ID",
                "publish_time",     
                "post_like",
                "post_commet",
                "post_score",
                "before_ID"
                ]
    user_features=features[:1]
    item_features=features[1:] 
    text_feature = ["post_ID",
                    "post_tag",
                    "post_topic",
                    "post_content"
                    ]
    df=pd.read_csv("data/data.csv")
    a=DataManager(df,features,user_features,item_features,text_feature=text_feature)
    # 
    a.init_text(text_feature)
    a.rebuild_data_negative_sample_pool(save_path="data/data2_ng.csv")
    
    print("hello,world")
