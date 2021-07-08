"""
Version:0.0.2 
Auther:Ace
Descirption:基础编码器,其他所以编码器都从此编码器中派生。
主要功能有：
1. 对特征进行编码和解码
2. 记录特征出现的位置
3. 
"""

from multiprocessing import set_forkserver_preload
import pickle
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import insert
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.generic_utils import default
from wrapt.wrappers import PartialCallableObjectProxy
import os
import sys
sys.path.append(".")
from featureConfig.feature1 import *
class EncoderManager2():
    name="Encoder"
    defaultPath=name+".pkl"
    def __init__(self,ids,lbePath="",dir="") -> None:
        self.defaultPath=os.path.join(dir,self.name+".pkl")
        if lbePath!="":
            self.lbe=pickle.load(lbePath)
        else:
            self.lbe=LabelEncoder()
        self.reset(ids)
    def reset(self,ids):
        # 对用户标记进行编码
        self.ids=self.lbe.fit_transform(ids)+1
        self.max_idx=self.ids.max()

    def transform(self,id):
        # 获取转换后的Id 
        if isinstance(id,list):
            return self.lbe.transform(id)
        else:
            return self.lbe.transform([id])[0]
    
    def re_transform(self,new_id):
        # 返回原始编码
        if isinstance(new_id,list):
            return self.lbe.inverse_transform(new_id)
        else:
            return self.lbe.inverse_transform([new_id])[0]
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


class EncoderManager():
    """
        改用dict实现
        """
    name="Encoder"
    defaultPath=name+".pkl"
    def __init__(self,ids,lbePath="",dir="") -> None:
        self.defaultPath=os.path.join(dir,self.name+".pkl")
        if lbePath!="":
            self.lbe=pickle.load(lbePath)
        else:
            self.word2idx=dict()
            self.idx2word=["error"]
            self.max_idx=1
        self.reset(ids)
    def __len__(self):
        return self.max_idx
    def __contains__(self,id):
        return id in self.word2idx

    def reset(self,ids):
        ids=list(set(ids))
        for one in ids:
            if one in self.word2idx:
                continue
            self.word2idx[one]=self.max_idx
            self.idx2word.append(one)
            self.max_idx+=1
            
    def get_words(self):
        return list(self.word2idx.keys())
    def transform(self,id):
        # 获取转换后的Id 
        if isinstance(id,list):
            
            return [self.word2idx[one] for one in id]
        else:
            return self.word2idx[id]
    
    def re_transform(self,new_id):
        # 返回原始编码
        if isinstance(new_id,list):
            return [self.idx2word[one] for one in new_id]
        else:
            return self.idx2word[new_id]
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
    def get_params(self):
        return self.word2idx,self.idx2word
    


class EncoderManagerPlus():
    """
        改用dict实现
        增加了历史行为记录功能
        支持的内置操作：len，in, 索引[]
        """
    name="Encoder"
    defaultPath=name+".pkl"
    def __init__(self,ids,dir="") -> None:
        self.defaultPath=os.path.join(dir,self.name+".pkl")
        self.word2idx=dict()
        self.history=dict()
        # id0 为默认的错误，id从1开始
        self.idx2word=["error"]
        self.max_idx=1
        self.reset(ids)
    def __len__(self):
        return self.max_idx
    def __contains__(self,id):
        return id in self.word2idx
        
    def reset(self,ids):
        ids=list(set(ids))
        self.type=type(ids[0])
        for one in ids:
            if one in self.word2idx:
                continue
            self.word2idx[one]=self.max_idx
            # 初始化记录表
            self.history[self.max_idx]=[]
            self.idx2word.append(one)
            self.max_idx+=1

    def record_history(self,dataset,feature1,feature2):
        """
        要求dataset为pd.DataFrame格式
        并且，是对uid和iid转换之后的数据
        """
        
        for user,group in dataset.groupby([feature1]):
            items=group[feature2].tolist()
            self.history[user].extend(items)
            
    def get_words(self):
        return list(self.word2idx.keys())

    def _transform(self,id):
        if id not in self.word2idx:
            raise "do not exist this word {}!".format(id)
        return self.word2idx[id]
    def transform(self,id):
        # 获取转换后的Id
        if isinstance(id,list) or isinstance(id,tuple):
            return [self._transform(one) for one in id]
        else:
            return self._transform(id)

    def __getitem__(self,id):
        return self.transform(id)

    def re_transform(self,new_id):
        # 返回原始编码
        if isinstance(new_id,list):
            return [self.idx2word[one] for one in new_id]
        else:
            return self.idx2word[new_id]
    
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
    def get_params(self):
        return self.word2idx,self.idx2word
    
    def get_history(self,user):
        if not isinstance(user,int):
            user=self.transform(user)
        return self.history[user]



    

if __name__=="__main__":
    a=EncoderManagerPlus([2,2,4])
    print(a[[1,4]])