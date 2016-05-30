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

def saveObject(filename,obj):
    f=open(filename,'wb')
    pickle.dump(obj,f)
    f.close()
    return True


def loadObject(filename):
    f=open(filename,'r')
    obj=pickle.load(f)
    return obj

etlregex = re.compile(ur"[^\u4e00-\u9f5aa-zA-Z0-9]")
def etl(content):
    content = etlregex.sub('',content)
    return content

#载入详情数据
docinfos = loadObject(output + "all.info")
#载入字典
dictionary = corpora.Dictionary.load(output + "all.dic")

#TF*IDF模型生成
#载入TFIDF模型
tfidfModel = models.TfidfModel.load(output+"allTFIDF.mdl")
indexTfidf = similarities.MatrixSimilarity.load(output + "allTFIDF.idx")

#载入LDA索引
ldaModel = models.LdaModel.load(output + "allLDA400Topic.mdl")
indexLDA = similarities.MatrixSimilarity.load(output + "allLDA400Topic.idx")




query= """
取舍之道，说起来容易，做起来却很难。有时候它意味着你要放弃自己擅长的、甚至给你带来成功的东西，而去在全新的、前景未知的领域做出尝试。如果柯达能早点从胶片中走出来、诺基亚能少造点塞班机，也许就不会有今天的佳能、苹果、三星。如果保时捷没有冒天下之大不韪造出卡宴，也很难预料它如今的境遇会如何。
看开头这应该是一篇为全新宝马X1 Li（下文简称新X1）洗地的文章，我想很多宝马死忠、车神也已经准备移步评论区，敲下神圣、激扬的文字。其实这年月已经不流行道德绑架了，所以对于车迷而言，也请适当的放下你对这个品牌的喜爱，试着从一个兜里揣着三十来万、想买辆豪华品牌SUV的消费者的角度来看待新X1，这也是本文作者的角度。
一切的争议其实都是由UKL前驱平台而起，宝马做了一个艰难的决定，给几个入门的车系换上了前驱平台。在空间、成本相对有限的入门车型里，放弃后驱传统以换取更大空间。我想这也能看出宝马对于未来趋势的判断，就是在难以兼顾空间、操控的入门豪华车市场，消费者会重视空间实用性多过操控性，宝马也将宝押在了空间实用性上。
宝马做出了取舍，消费者该怎么选？以我自己为例吧，在买车的时候也考虑过老款宝马X1（下文简称老X1），优惠后25万左右，价格没比途观、奇骏贵多少，但品牌、动力的提升都可谓巨大。但为什么最后放弃？正是因为空间。如果家里只有这一辆车，老X1的空间确实有些力不从心。
新X1的空间绝对没有问题，而且看起来也比老款要更大气、更有面子。我们测试的这台xDrive25Li 豪华型同样采用2.0T和8AT变速箱，只是改为基于前驱的四驱系统。究竟值不值得选择，读完文章希望你能有答案。
更大气、更阳刚新X1在外观上的变化翻天覆地，与之前的老X1是完全性格的两种产物。老X1低调内敛，有着一种含蓄之美；而新X1则阳刚帅气，骨子里透着一种坚韧的性格，有着更符合男性消费者需求的阳刚之美。
"""
query_bow = dictionary.doc2bow(filter(lambda x: len(x)>0,map(etl,jieba.cut(query,cut_all=False))))
tfidfvect = tfidfModel[query_bow]
simstfidf = indexTfidf[tfidfvect]
sort_sims = sorted(enumerate(simstfidf), key=lambda item: -item[1])
print "TFIDF similary Top 10:::"
for sim in sort_sims[:10]:
    print "ID : " + docinfos[sim[0]]["id"] + "\t" + docinfos[sim[0]]["title"] + "\tsimilary:::" + str(sim[1])


ldavec = ldaModel[tfidfvect]
simlda = indexLDA[ldavec]
sort_sims = sorted(enumerate(simlda), key=lambda item: -item[1])
print "LDA similary Top 10:::"
for sim in sort_sims[:10]:
    print "ID : " + docinfos[sim[0]]["id"] + "\t" + docinfos[sim[0]]["title"] + "\tsimilary:::" + str(sim[1])





query= """
一般情况下，搜索引擎默认会认为索引是不会有太大的变化的，所以把索引分为全量索引和增量索引两部分，全量索引一般是以天甚至是周，月为单位构建的，构建完了以后就导入到引擎中进行检索，而增量索引是实时的进入搜索引擎的，很多就是保存在内存中，搜索的时候分别从全量索引和增量索引中检索数据，然后把两部分数据合并起来返回给请求方，所以增量索引不是我们这一篇的主要内容，在最后我的索引构建部分我会说一下我的增量索引构建方式。现在先看看全量索引
"""
query_bow = dictionary.doc2bow(filter(lambda x: len(x)>0,map(etl,jieba.cut(query,cut_all=False))))
tfidfvect = tfidfModel[query_bow]
simstfidf = indexTfidf[tfidfvect]
sort_sims = sorted(enumerate(simstfidf), key=lambda item: -item[1])
print "TFIDF similary Top 10:::"
for sim in sort_sims[:10]:
    print "ID : " + docinfos[sim[0]]["id"] + "\t" + docinfos[sim[0]]["title"] + "\tsimilary:::" + str(sim[1])


ldavec = ldaModel[tfidfvect]
simlda = indexLDA[ldavec]
sort_sims = sorted(enumerate(simlda), key=lambda item: -item[1])
print "LDA similary Top 10:::"
for sim in sort_sims[:10]:
    print "ID : " + docinfos[sim[0]]["id"] + "\t" + docinfos[sim[0]]["title"] + "\tsimilary:::" + str(sim[1])




