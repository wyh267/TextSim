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
import gensim


reload(sys);
sys.setdefaultencoding('utf8');



model = gensim.models.Word2Vec.load("./output/allw2v.w2v")  
sims=model.most_similar(u'全栈')
for sim in sims:
    print sim[0] + "\t" + str(sim[1])

