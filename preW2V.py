
# -*- coding: utf-8 -*-  
#
# 输入doc，生成分词后的语料
#
#
import re
import logging
import os.path
import sys
import multiprocessing

import jieba

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence



reload(sys)
sys.setdefaultencoding('utf8')
output = "./output/"

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
    word_list = filter(lambda x: len(x)>0,map(etl,jieba.cut(content,cut_all=False)))
    for w in word_list:
        print w + " ",
    print  ""
f.close()

