import sys
sys.path.append(".")
from nltk.featstruct import FeatStructReader
from DataManager.encoderManager import EncoderManager,EncoderManagerPlus
from DataManager.featureManager import FeatureManagerPlus
from featureConfig.feature1 import *
import pickle
from tools import transFeature


class UserManager(EncoderManager):
    name="UserManager"
    pass

class UserManagerPlus(EncoderManagerPlus):
    name="UserManager"
    pass


class UserManagerFeature():
    """
        改用dict实现
        增加了历史行为记录功能
        """
    name="Manager"
    defaultPath=name+".pkl"
    def __init__(self,df,features,dir="") -> None:
        self.defaultPath=os.path.join(dir,self.name+".pkl")
        # 特征管理器
        self.data=df
        self.features=features
        self.features_managers=dict()
        self.reset(df,features)
    def reset(self,df,features):
        # 第一个特征为用户特征
        uids=df[features[0]].unique().tolist()
        self.features_managers[features[0]]=UserManager(uids)
        for feature in features[1:]:
            ids=df[feature].unqiue().tolist()
            self.features_managers[feature]=FeatureManagerPlus(ids)
        # 重新调整编码
        for feature, manager in self.features_managers.items():
            transFeature(self.data,feature,manager)
        # 将用户id作为索引，方便查询
        self.data.set_index(features[0])

    def get_features(self,user):
        uid=self.features_managers[self.features[0]].transform(user)
        return self.data.loc[uid]

    def reload(self,path=""):
        if path=="":
            path=self.defaultPath
        with open(path,"rb") as f:
            # self.lbe=pickle.load(f)
            self=pickle.load(f)
    def store(self,path=""):
        if path=="":
            path=self.defaultPath
        with open(path,"wb") as f:
            # pickle.dump(self.lbe,f)
            pickle.dump(self,f)


    pass
if __name__=="__main__":
    pass

