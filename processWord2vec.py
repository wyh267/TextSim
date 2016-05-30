#!/bin/python
#encoding=utf8
import re
import logging
import os.path
import sys
import multiprocessing

import jieba

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence



reload(sys);
sys.setdefaultencoding('utf8');

etlregex = re.compile(ur"[^\u4e00-\u9f5aa-zA-Z0-9]")
def etl(content):
    content = etlregex.sub('',content)
    return content



model = Word2Vec(LineSentence('./data/allw2v.txt'),size=10000,window=5,min_count=5,workers=multiprocessing.cpu_count())

model.save('./output/allw2v.w2v')
#model.save_word2vec_format('segmentfaultw2v.txt',binary=False)