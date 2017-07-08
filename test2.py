# -*- coding: utf-8 -*-
'''
Created on 2016年6月3日

@author: sx
'''
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer 
import jieba
import pandas
from textProcess import textProcess
import jieba.analyse
if __name__=="__main__":
    """
    searchData=['H04M1/12',
                         'H04L29/08',
                         'A61C1/08',
                         'A61B5/0245',
                         'A61B5/11',
                         'G08B25/10',
                         'H04N7/18',
                         'A63F13/24',
                         'A63F13/90',
                         'H04M1/02',
                         'G09B7/02',
                         'F24D17/02',
                         'F24D19/10',
                         'H02J13/00',
                         'H04M1/11']
    patCls=searchData[1].decode("utf-8").encode("utf-8")
    pt= patCls.replace('/','-')
    print pt
    print searchData[0].decode("utf-8")
    print type(searchData[1])
    
    """
    
    
    
    rawdata="在现有技术中移动终端与遥控器之间通常采用无线连接的方式，但采用无线连接容易受到信号的干扰，而使得数据传输存在很多不稳定因素，因此本发明的移动终端与遥控器之间采用USB-OTG接口，与现有技术中采用无线连接相比，数据传输更加稳定；该移动终端具有上网功能，并通过共享网络对第一路由器和第二路由器分配IP地，使得无人机能够直接访问Internet。"
    
    rawdata1="在现有技术中移动终端与遥控器之间通常采用无线连接的方式，但采用无线连接容易受到信号的干扰，而使得数据传输存在很多不稳定因素，因此本发明的移动终端与遥控器之间采用USB-OTG接口，与现有技术中采用无线连接相比，数据传输更加稳定；该移动终端具有上网功能，并通过共享网络对第一路由器和第二路由器分配IP地址，使得无人机能够直接访问Internet。"
    tex=textProcess().tokenProcess(rawdata)
    print tex
    for text in tex: 
        print text
    """
    texts_tokenized=list(jieba.cut(rawdata))
    texts_filtered_stopwords=[]
    for i in texts_tokenized: 
        i=i.encode('utf-8')
        print i            
        texts_filtered_stopwords.append(i)
    print "*****************************************************"
    #去除标点符号
    english_punctuations = ['，',',', '。','.', '：',';', '；',':', '？','?', '（', '）', '【', '】', '&', '！', '*', '@', '#', '￥', '%','、','/','(',')','[',']','!','$']
    texts_filtered = [word for word in texts_filtered_stopwords if word not in  english_punctuations]
    print texts_filtered
    print "去除标点后："
    st = LancasterStemmer()
    texts_stemmed = [st.stem(word) for word in  texts_filtered]
    print "词干化"
    """