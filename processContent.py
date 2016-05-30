#!/bin/python
#encoding=utf8

from gensim import corpora,models,similarities,utils
import jieba
import jieba.posseg as pseg
import sys
import os
import re
import gc
import pickle
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

reload(sys)
sys.setdefaultencoding('utf8')
output = "./output/"

jieba.load_userdict( "user_dic.txt" )

def saveObject(filename,obj):
    f=open(filename,'wb')
    pickle.dump(obj,f)
    f.close()
    return True

etlregex = re.compile(ur"[^\u4e00-\u9f5a0-9]")
def etl(content):
    content = etlregex.sub('',content)
    return content
    
    
#原始语料集合
train_set=[]
docinfos = []
#读取文本，进行切词操作
f=open("./data/all.txt")
lines=f.readlines()
for line in lines:
    content = (line.lower()).split("\t")[2] + (line.lower()).split("\t")[1]
    
    
    #print content
    #切词，etl用于去掉无用的符号
    word_list = filter(lambda x: len(x)>0,map(etl,jieba.cut(content,cut_all=False)))
    train_set.append(word_list)
    detail={}
    detail["id"]=(line.lower()).split("\t")[0]
    detail["title"]=(line.lower()).split("\t")[1]
    detail["content"]=(line.lower()).split("\t")[2]
    docinfos.append(detail)
f.close()
#语料太大的情况下可以强制GC回收内存空间
#gc.collect()
#生成字典
dictionary = corpora.Dictionary(train_set)
#去除极低频的杂质词
dictionary.filter_extremes(no_below=1,no_above=1,keep_n=None)
#将词典保存下来，将语料也保存下来,语料转换成bow形式，方便后续使用
dictionary.save(output + "all.dic")
corpus = [dictionary.doc2bow(text) for text in train_set]
saveObject(output+"all.cps",corpus)
#存储原始的数据
saveObject(output+"all.info",docinfos)

#TF*IDF模型生成
#使用原始数据生成TFIDF模型
tfidfModel = models.TfidfModel(corpus)
#通过TFIDF模型生成TFIDF向量
tfidfVectors = tfidfModel[corpus]
#存储tfidfModel
tfidfModel.save(output + "allTFIDF.mdl")
indexTfidf = similarities.MatrixSimilarity(tfidfVectors)
indexTfidf.save(output + "allTFIDF.idx")


#LDA模型
lda = models.LdaModel(tfidfVectors, id2word=dictionary, num_topics=50)
lda.save(output + "allLDA50Topic.mdl")
corpus_lda = lda[tfidfVectors]
indexLDA = similarities.MatrixSimilarity(corpus_lda)
indexLDA.save(output + "allLDA50Topic.idx")



