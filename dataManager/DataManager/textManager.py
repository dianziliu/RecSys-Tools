from io import StringIO
import os
import pickle
import re
import sys
from tokenize import group
import requests
from nltk import data, text
import json
sys.path.append(".")

import pickle
import jieba
import nltk
import numpy as np
import tensorflow as tf
from featureConfig.feature1 import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from DataManager.encoderManager import EncoderManager,EncoderManagerPlus
from DataManager.userManager import UserManager
from DataManager.itemManager import ItemManager

from tqdm import tqdm


nltk.data.path.append("./nltk_data")

class WordsManager(EncoderManagerPlus):
    name="WordsManager"

    def downloadEmbedding(self):
        words=self.get_words()
        self.emb=download(words)
        pass

def download(word):
    """ 腾讯词向量，空格分割多个单词
        """
    url_format="http://ai-net.flyai.com/ai?project_id=cece_service_tencent_vectors&online=1&ab=2&prefix=cece&user_id=xxx&seg={}"
    max_len=100

    embs=dict()
    if isinstance(word,list):
        len_input=len(word)
        start=0
        while start<len_input:
            sub_words=word[start:start+max_len]
            url = url_format.format(" ".join(sub_words))
            r=requests.get(url)
            r.raise_for_status()
            r.encoding=r.apparent_encoding
            # print(r.text)
            this_emb=json.loads(r.text)
            embs.update(this_emb)
            start+=max_len
    return embs


class TextManager():
    """
    用来管理数据中的文本信息
    """
    stopwords = nltk.corpus.stopwords.words('english')
    
    def clean_str(self,string):
        if not isinstance(string,str):
            return ""
        string=re.sub(r"[,.!?:;，。！？：；]","",string)
        return string.strip()

    def get_vocab_freq(self,corpus, vocab_size=20000):
        # 构建词典
        freq_dict = {}
        for text in corpus:
            if not isinstance(text,str):
                continue
            # tokens = nltk.word_tokenize(text)
            text=self.clean_str(text)
            tokens=jieba.lcut(text)
            for token in tokens:
                freq_dict[token] = freq_dict.get(token, 0)+1
        tuples = [[v[1],v[0]] for v in freq_dict.items()]
        tuples = sorted(tuples)[::-1]
        vocab = []
        for tp in tuples:
            word = tp[1]
            if word not in self.stopwords:
                vocab.append(word)
                if len(vocab)>20000:
                    break
        return vocab


    def get_vocab_tfidf(self,corpus, vocab_size = 20000):
        
        tfidf_model = TfidfVectorizer().fit(corpus)
        sparse_result = tfidf_model.transform(corpus)
        mat = np.array(sparse_result.todense())
        print(mat.shape)
        vec = np.max(mat, axis=0)
        print(vec.shape)
        index = np.argsort(vec)[::-1]
        vocab = []
        # total_vocab_size = min(vocab_size, len(tfidf_model.vocabulary_.keys()))
        idx2word = {v:k for k,v in tfidf_model.vocabulary_.items()}

        for idx in index:
            word = idx2word[idx]
            if word not in self.stopwords:
                vocab.append(word)
                if len(vocab)>vocab_size:
                    break
        return vocab

    def vectorize_texts(self,texts,word2idx):
        # 将原始文本向量化
        vec_texts = [[0]]

        # 需要优化
        for text in tqdm(texts,ncols=100):
            if not isinstance(text,str):
                continue
            text=self.clean_str(text)
            tokens=jieba.lcut(text)
            # 需要过滤掉一些词
            # tokens=[token for token in tokens if token in self.vocab]
            # vec_text=self.words_manager.transform(tokens)
            vec_text = [word2idx[token] for token in tokens if token in word2idx]
            

            vec_texts.append(vec_text)
        return vec_texts
    
    def build_vocab(self,data_df):

        # 读取数据集
        
        # df=data_df.drop_duplicates(item_features[0])
        texts=[]
        # 获取文本数据
        for index,row in tqdm(data_df.iterrows(),ncols=100):
            text_data=""
            for feature in text_features:
                text_data+=self.clean_str(row[feature])
            texts.append(text_data)
            # 词典
        self.vocab=self.get_vocab_freq(texts)

        # 编码解码
        # word2idx = {word:i+1 for i,word in enumerate(vocab)}
        # idx2word = {i+1:word for i,word in enumerate(vocab)}

        # 对字符进行编码
        self.words_manager=WordsManager(self.vocab)
        #
        word2idx,idx2word=self.words_manager.get_params()

        # word2idx = {word: idx for word, idx in zip(
        #     self.vocab, self.words_manager.transform(self.vocab))}

        # 对文本进行编码
        vec_texts = self.vectorize_texts(texts,word2idx)

        lens = [len(text) for text in vec_texts]
        print(max(lens), min(lens), np.mean(lens))

        # print(vocab)
        return vec_texts,self.vocab,word2idx

    def build_vocab_only(self,data_df,item_manager):
        """
            对帖子进行唯一编码
        """
        # 读取数据集
        
        # 去重
        df=data_df.drop_duplicates(item_features[0])

        # 需要给空的帖子进行预留
        texts=[""]*item_manager.max_idx

        # 获取文本数据
        for index, row in df.iterrows():
            iid=row[item_features[0]]
            text_data=""
            for feature in text_features:
                text_data+=self.clean_str(row[feature])
            texts[iid]+=text_data
            # texts[item_manager.transform(iid)]+=text_data
        # 词典
        self.vocab=self.get_vocab_freq(texts)

        # 对字符进行编码
        self.words_manager=WordsManager(self.vocab)

        word2idx,idx2word=self.words_manager.get_params()

        # 对文本进行编码
        vec_texts = self.vectorize_texts(texts,word2idx)

        lens = [len(text) for text in vec_texts]
        print(max(lens), min(lens), np.mean(lens))

        # print(vocab)
        return vec_texts,self.vocab,word2idx


    def build_data(self,path):
        df=pd.read_csv(path)
        self.build_vocab(df)


    def build_vocab_test(self,path):
        data_df=pd.read_csv(path)
        # 读取数据集
        
        # df=data_df.drop_duplicates(item_features[0])
        texts=[]
        # 获取文本数据
        for index,row in tqdm(data_df.iterrows(),ncols=100):
            text_data=""
            for feature in text_features:
                text_data+=self.clean_str(row[feature])
            texts.append(text_data)
            # 词典
        self.vocab=self.get_vocab_freq(texts)

        # 编码解码
        # word2idx = {word:i+1 for i,word in enumerate(vocab)}
        # idx2word = {i+1:word for i,word in enumerate(vocab)}

        # 对字符进行编码
        self.words_manager=WordsManager(self.vocab)
        self.words_manager.downloadEmbedding()


class TextManagerPlus():
    """
    用来管理数据中的文本信息
    """
    stopwords=[]
    def clean_str(self,string):
        if not isinstance(string,str):
            return ""
        string=re.sub(r"[,.!?:;，。！？：；]","",string)
        return string.strip()

    def get_vocab_freq(self,corpus, vocab_size=20000):
        # 构建词典
        freq_dict = {}
        for text in corpus:
            if not isinstance(text,str):
                continue
            # tokens = nltk.word_tokenize(text)
            text=self.clean_str(text)
            tokens=jieba.lcut(text)
            for token in tokens:
                freq_dict[token] = freq_dict.get(token, 0)+1
        tuples = [[v[1],v[0]] for v in freq_dict.items()]
        tuples = sorted(tuples)[::-1]
        vocab = []
        for tp in tuples:
            word = tp[1]
            if word not in self.stopwords:
                vocab.append(word)
                if len(vocab)>vocab_size:
                    break
        return vocab


    def get_vocab_tfidf(self,corpus, vocab_size = 20000):
        
        tfidf_model = TfidfVectorizer().fit(corpus)
        sparse_result = tfidf_model.transform(corpus)
        mat = np.array(sparse_result.todense())
        print(mat.shape)
        vec = np.max(mat, axis=0)
        print(vec.shape)
        index = np.argsort(vec)[::-1]
        vocab = []
        # total_vocab_size = min(vocab_size, len(tfidf_model.vocabulary_.keys()))
        idx2word = {v:k for k,v in tfidf_model.vocabulary_.items()}

        for idx in index:
            word = idx2word[idx]
            if word not in self.stopwords:
                vocab.append(word)
                if len(vocab)>vocab_size:
                    break
        return vocab

    def vectorize_texts(self,texts,word_manager):
        # 将原始文本向量化
        vec_texts = dict()

        # 需要优化
        for iid,text in tqdm(texts.items(),ncols=100):
            if not isinstance(text,str):
                continue
            text=self.clean_str(text)
            tokens=jieba.lcut(text)
            # 需要过滤掉一些词
            # tokens=[token for token in tokens if token in self.vocab]
            # vec_text=self.words_manager.transform(tokens)
            vec_text = [word_manager[token] for token in tokens if token in word_manager]
            # vec_texts.append(vec_text)
            vec_texts[iid]=vec_text
        return vec_texts
    
    def build_vocab(self,data_df,text_features,word_manager=None):
        """
            对帖子进行唯一编码
            """
        # 读取数据集
        
        # 去重
        df=data_df.drop_duplicates(text_features[0])

        # 需要给空的帖子进行预留
        src_texts=dict()

        # 获取文本数据
        for index,row in df.iterrows():
            iid=row[item_features[0]]
            text_data=""
            for feature in text_features:
                text_data+=self.clean_str(row[feature])
            src_texts[iid]=text_data
        self.src_texts=src_texts
        # 词典
        self.vocab=self.get_vocab_freq(src_texts.values())

        if word_manager is None:
        # 对字符进行编码
            self.words_manager=WordsManager(self.vocab)
        else:
            self.words_manager=word_manager

        # 对文本进行编码
        self.vec_texts = self.vectorize_texts(src_texts,self.words_manager)

        lens = [len(text) for text in self.vec_texts.values()]
        print(max(lens), min(lens), np.mean(lens))

        # print(vocab)
        return self.vec_texts,self.vocab,self.words_manager

    def build_data(self,path):
        df=pd.read_csv(path)
        features=[
            "post_ID",
            "post_tag",
            "post_topic",
            "post_content"
        ]
        self.build_vocab(df,features)


if __name__ == '__main__':
    # a=WordsManager([1,2,3])
    # a.download(["你好","我","他"])
    a=TextManagerPlus()
    a.build_data("rsData/test4.csv")
    # a.build_vocab_test("rsData/test4.csv")