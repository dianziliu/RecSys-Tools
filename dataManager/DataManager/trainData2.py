from itertools import count
import os
import pickle
import re
import sys

from nltk import text
from nltk.util import ngrams
from numpy.lib.npyio import save

sys.path.append(".")

import pickle
from random import choices
import nltk
import numpy as np
from featureConfig.feature1 import *
from tqdm import tqdm
from scipy import sparse

from DataManager.encoderManager import EncoderManager
from DataManager.tools import transData2Frame
from DataManager.itemManager import ItemManager,ItemManagerPlus
from DataManager.textManager import TextManager
from DataManager.userManager import UserManager,UserManagerPlus
from utility.decorator import logger
nltk.data.path.append("./nltk_data")

REUBID=True


class TrainData():
    """
    用于维护训练数据集
    """
    def __init__(self,dir,user_manager,item_manager,text_manager,*args,**kwargs) -> None:
        """
            dir: 存在中间文件的路径
            """
        self.dir=dir
        self.user_manager = user_manager
        self.item_manager = item_manager
        self.text_manager = text_manager

        # 用于负采样
        self.items=set([i for i in range(1,item_manager.max_idx)])

    def vectorize_ui_text(self,ui_manager,ui_history):
        # 对历史记录进行编码
        vec_ui_text = []
        err_count=0
        for idx in range(1,ui_manager.max_idx):
            ui=idx
            # 获取历史记录
            if not ui in ui_history:
                err_count+=1
                continue
            text_idx=ui_history[ui]
            vec_ui_text.append(np.random.permutation(text_idx))
        print("miss {} id".format(err_count))
        return vec_ui_text
    
    def negative_sample(self,history):
        # 下一步，使用之前看到的代替（）
        return choices(list(self.items-set(history)), k=len(history))

    def build_ui_vocab(self,data_df):
        u_dict = {}
        i_dict = {}
        # 遍历数据
        for i,line in tqdm(data_df.iterrows(),ncols=100):
            # user = line['reviewerID']
            # item = line['asin']
            user=line[user_features[0]]
            item=line[item_features[0]]
            # 
            if user not in u_dict:
                u_dict[user] = [item]
            else:
                u_dict[user].append(item)
            if item not in i_dict:
                i_dict[item] = [item]
            else:
                if user in u_dict and len(u_dict[user])>1:
                    i_dict[item].append(u_dict[user][1])
                else:
                    pass

        vec_u_text = self.vectorize_ui_text(self.user_manager, u_dict)
        vec_i_text = self.vectorize_ui_text(self.item_manager, i_dict)

        vec_uit = []

        for idx,row in tqdm(data_df.iterrows(),ncols=100):
            uid=row[user_features[0]]
            iid=row[item_features[0]]
            vec_uit.append((uid, iid,iid,1))
            beforeIDs=row[neg_feature[0]]
            for j in beforeIDs:
                vec_uit.append((uid,j,j,-1))            

        return vec_uit, vec_u_text, vec_i_text  


    def build_ui_vocab_only(self,data_df):
        # 暂时没用了
        u_dict = {}
        i_dict = {}
        # 遍历数据
        for i,line in tqdm(data_df.iterrows(),ncols=100):
            # user = line['reviewerID']
            # item = line['asin']
            user=line[user_features[0]]
            item=line[item_features[0]]
            idx = i+1
            if user not in u_dict:
                u_dict[user] = [idx]
            else:
                u_dict[user].append(idx)
            if item not in i_dict:
                i_dict[item] = [idx]
            else:
                i_dict[item].append(idx)
        vec_u_text = self.vectorize_ui_text(self.user_manager, u_dict)
        vec_i_text = self.vectorize_ui_text(self.item_manager, i_dict)

        vec_uit = []
        text_idx=0

        # # 实现了 构建标准的训练用数据集
        # for user,group in tqdm(data_df.groupby(user_features[0]),ncols=100):
        #     # u=self.user_manager.transform(user)
        #     u=user
        #     sub_items=group[item_features[0]].tolist()
        #     for i in self.item_manager.transform(sub_items):
        #         vec_uit.append((u, i, text_idx,1))
        #         text_idx = text_idx+ 1
        #     beforeIDs=group[neg_feature[0]]
        #     ngs = self.negative_sample(sub_items)
        #     for j in ngs:
        #         vec_uit.append((u, i,-1,0))

        # 翻转数据 速度快
        users=data_df[user_features[0]].tolist()
        # users=self.user_manager.transform(users)
        items = data_df[item_features[0]].tolist()
        # items=self.item_manager.transform(items)

        df_vec_uit=pd.DataFrame({
            "uid":users,
            "iid":items,
            "text_idx":items
        })

        # df=pd.concat([data_df,df_vec_uit])
        df=data_df.join(df_vec_uit)
        for user,group in tqdm(df.groupby(user_features[0]),ncols=100):
            # u=self.user_manager.transform(user)
            uids=group.uid.tolist()
            sub_items=group.iid.tolist()
            text_idx=group.text_idx.tolist()
            for uid,iid,idx in zip(uids,sub_items,text_idx):
                vec_uit.append((uid, iid,idx,1))
            # ngs = choices(list(self.items-set(sub_items)), k=len(sub_items))
            
            # 来自beforeID字段的负样本 
            beforeIDs=group[neg_feature[0]].tolist()
            ngs=[]
            for one in beforeIDs:
                if not isinstance(one,str):
                    continue
                ngs.extend(
                    one.split(",")
                )
            # ngs=self.item_manager.transform(ngs)
            for tmp_j in ngs:
                # tmp_j=self.item_manager.transform(j)
                vec_uit.append((uids[0],tmp_j,tmp_j, -1))
            # 随机负采样
            ngs=self.negative_sample(sub_items)
            for j in ngs:
                vec_uit.append((uids[0], j,j, -1))


        return vec_uit, vec_u_text, vec_i_text  

    def get_neighbors_1(self,mat):
        """
        计算邻域相似度
        """
        rows,cols = mat.shape
        # 爆内存
        # TODO 改用稀疏矩阵
        res_mat = np.zeros((rows,rows))

        for i in range(rows):
            sys.stdout.write('\r {}/{}'.format(i,rows))
            a1 = mat[i]
            nz_idx = np.where(a1!=0)[0]
            a1 = a1[nz_idx]
            sub_mat = mat[i:].T[nz_idx].T

            commo_vec = (a1*sub_mat!=0).astype(int)
            equal_vec = (a1==sub_mat).astype(int)

            coeq_vote = np.sum(commo_vec*equal_vec, axis=1)
            comm_vote = np.sum(commo_vec, axis=1)

            norm = np.sqrt(comm_vote)
            norm = np.maximum(norm, 1)
            res = coeq_vote/norm 

            res_mat[i,i:] = res 
            res_mat[i:,i] = res 
        return res_mat


    def get_doc_neighbors(self,vec_uit, u_neighbor_mat, i_neighbor_mat, name='item'):

        # 根据相似度选择最相似的num个邻居
        uit = np.array(vec_uit)
        u_dat = uit[:,0]
        i_dat = uit[:,1]
        d_dat = uit[:,2]
        # y_dat = uit[:,3]
        num = 10

        data1 = i_dat if name == 'item' else u_dat
        data2 = u_dat if name == 'item' else i_dat
        neighbor_mat = u_neighbor_mat if name == 'item' else i_neighbor_mat
        res = []
        length = len(uit)
        for i in range(length):
            # label = y_dat[i]
            label=1
            item = data1[i]

            # 返回item的历史记录的索引
            index = np.where(data1==item)[0]

            # labels = y_dat[index]

            # index = index[labels == label]

            # index = np.where(y_dat == label)[0]
            # IndexError: index 6105 is out of bounds for axis 0 with size 5082
            
            # 返回user的历史记录的索引
            user = data2[i]
            users = data2[index]

            # 计算users的相似度
            sims = neighbor_mat[user]
            similarity = sims[users]

            # 对相似度进行排序
            sort_index = np.argsort(similarity)[::-1]

            # 不够补0
            addition = max(num-len(index), 0)
            selection = len(index) if num>len(index) else num

            # 
            neighbors_1 = np.concatenate([index[sort_index][:selection], [0]*addition])
            neighbors_1[0] = 0
            neighbors_1 = np.random.permutation(neighbors_1)
            
            # 将索引映射到文本序号
            neighbors_1=[uit[int(one),2] for one in neighbors_1]

            # if len(index)-1>=num:
            # 	neighbors_1 = index[sort_index][1:num+1]+1
            # else:
            # 	neighbors_1 = np.concatenate([index[sort_index][1:]+1,np.array([i+1]*(num-len(index)+1))])

            res.append(neighbors_1)

        return np.array(res)

    
    def build_ui_text(self,vec_uit):
        num_user=self.user_manager.max_idx
        num_item=self.item_manager.max_idx
        
        filename=os.path.join(self.dir,"uitext")
        # 默认为0，因此 负样本为-1
        ui_mat = np.zeros((num_user, num_item))
        pmtt_file = filename+'_pmtt.npy'
        if os.path.exists(pmtt_file) and (not REUBID):
            pmtt = np.load(pmtt_file)
        else:
            pmtt = np.random.permutation(len(vec_uit))

        train_size = int(len(vec_uit)*0.8)
        train_vec_uit = np.array(vec_uit)[pmtt][:train_size]

        for uit in train_vec_uit:
            u = uit[0]
            i = uit[1] 
            r = uit[3]
            ui_mat[u,i] = r 
        # u_neighbor_mat = np.zeros((num_user, num_user))
        file = filename+'_u_neighbors.npy'
        if (not os.path.exists(file)) or REUBID:
            print('user neighbors file not exists')
            u_neighbor_mat = self.get_neighbors_1(ui_mat)
            np.save(file, u_neighbor_mat)
        else:
            u_neighbor_mat = np.load(file)
        file = filename+'_i_neighbors.npy'
        if (not os.path.exists(file)) or REUBID:
            print('user neighbors file not exists')
            i_neighbor_mat = self.get_neighbors_1(ui_mat.T)
            np.save(file, i_neighbor_mat)
        else:
            i_neighbor_mat = np.load(file)

        print("getting documents...")
        vec_u_text = self.get_doc_neighbors(vec_uit, u_neighbor_mat, i_neighbor_mat, 'item')
        vec_i_text = self.get_doc_neighbors(vec_uit, u_neighbor_mat, i_neighbor_mat, 'user')
        np.savetxt(os.path.join(self.dir,'utexts.txt'), vec_u_text, fmt='%d')
        np.savetxt(os.path.join(self.dir,'itexts.txt'), vec_i_text, fmt='%d')
        return vec_u_text, vec_i_text
    
    def build_data(self,df):
        print('building ui vocab...')
        vec_uit, vec_u_text, vec_i_text =self.build_ui_vocab(df)
        print('building ui similar text...')
        vec_u_text, vec_i_text = self.build_ui_text(vec_uit)

        return vec_uit,vec_u_text,vec_i_text

def filterFalseNegSample(data,user_manager,item_manager):
    """
        根据用户的历史记录过滤掉假负样本
        """
    items=[]
    all_count=0
    count=0
    for idx,row in tqdm(data.iterrows(),ncols=100):
        user=row[user_features[0]]
        history=user_manager.get_history(user)
        one= row[neg_feature[0]]
        if isinstance(one,str):
            ngs=item_manager.transform(one.split(","))      
            tngs=[]
            for one in ngs:
                if one not in history:
                    tngs.append(one)
                else:
                    count+=1

            items.append(tngs)
        else:
            items.append([0])
    data[neg_feature[0]]=items
    print("user wated {} posts".format(count))
    # for user,group in data.groupby([user_features[0]]):
    #     items=[]
    #     history=user_manager.get_history(user)
    #     for one in group[neg_feature[0]].tolist():
    #         if isinstance(one,str):   
    #             ngs=item_manager.transform(one.split(","))
    #             tngs=[]
    #             for one in ngs:
    #                 if one not in history:
    #                     tngs.append(one)
    #             items.append(tngs)
    #         else:
    #             items.append([0])
    #     group[neg_feature[0]]=items



def buildTrainData(path,save_path=""):

    # 读取数据
    df=pd.read_csv(path)

    """这三类都可以独立实现，并进行传递"""
    # 建立用户管理
    users=df[user_features[0]].unique().tolist()
    user_manager = UserManager(users)
    user2idx,idx2user=user_manager.get_params()
    # user2idx = {word: idx for word, idx in zip(
    #     users, user_manager.transform(users))}

    # 物品管理
    items = df[item_features[0]].unique().tolist()
    for one in df[neg_feature[0]].tolist():
        if not isinstance(one,str):
            continue
        items.extend(one.split(","))
    item_manager = ItemManager(items)
    item2idx,idx2item=item_manager.get_params()
    # item2idx = {word: idx for word, idx in zip(
    #     items, item_manager.transform(items))}

    # 文本管理
    text_manager = TextManager()

    vec_texts, vocab, word2idx=text_manager.build_vocab(df)

    # 构建数据集
    train_data = TrainData(save_path,
                           user_manager,
                           item_manager,
                           text_manager)
    vec_uit,vec_u_text,vec_i_text = train_data.build_data(df)

    data_save = dict()
    data_save['vec_texts'] 	= vec_texts
    data_save['vocab'] 		= vocab
    data_save['word2idx'] 	= word2idx
    data_save['vec_uit'] 	= vec_uit
    data_save['vec_u_text'] = vec_u_text
    data_save['vec_i_text'] = vec_i_text
    data_save['user2idx'] 	= user2idx
    data_save['item2idx'] 	= item2idx

    print('writing back...')

    with open(os.path.join(save_path, 'data_save.pkl'), 'wb') as fr:
        pickle.dump(data_save, fr)

@logger("start to buid traing data")
def buildTrainDataRebuild(df,save_path=""):

    #-------------------
    # 整体数据的处理
    #-------------------

    # 读取数据
    # df=pd.read_csv(path)

    if save_path!="" and (not os.path.exists(save_path)):
        os.mkdir(save_path)
        
    """这三类都可以独立实现，并进行传递"""
    # 建立用户管理
    users=df[user_features[0]].unique().tolist()
    user_manager = UserManagerPlus(users)
    user2idx,idx2user=user_manager.get_params()

    # 物品管理
    items = df[item_features[0]].unique().tolist()
    for one in df[neg_feature[0]].tolist():
        if not isinstance(one,str):
            continue
        items.extend(one.split(","))
    item_manager = ItemManagerPlus(items)
    item2idx,idx2item=item_manager.get_params()

    # 转化uid和iid
    transData2Frame(df,user_manager,item_manager)

    # 重建数据表格
    user_manager.record_history(df,user_features[0],item_features[0])
    item_manager.record_history(df,item_features[0],user_features[0])
    # 处理before_ID字段
    filterFalseNegSample(df,user_manager,item_manager)

    # 构建单词表
    text_manager = TextManager()
    vec_texts, vocab, word2idx=text_manager.build_vocab_only(df,item_manager)

    #-------------------
    # 可以在这里插入数据划分的代码 
    #-------------------

    # 构建数据集
    train_data = TrainData(save_path,
                           user_manager,
                           item_manager,
                           text_manager)
    vec_uit,vec_u_text,vec_i_text = train_data.build_data(df)

    data_save = dict()
    data_save['vec_texts'] 	= vec_texts
    data_save['vocab'] 		= vocab
    data_save['word2idx'] 	= word2idx
    data_save['vec_uit'] 	= vec_uit
    data_save['vec_u_text'] = vec_u_text
    data_save['vec_i_text'] = vec_i_text
    data_save['user2idx'] 	= user2idx
    data_save['item2idx'] 	= item2idx

    print('writing back...')

    with open(os.path.join(save_path, 'data_save.pkl'), 'wb') as fr:
        pickle.dump(data_save, fr)


if __name__=="__main__":
    # build_train_data("rsData/test4.csv","tmp_save")
    origin_data=pd.read_csv("data/data.csv")
    buildTrainDataRebuild(origin_data,"tmp_save3")
    pass