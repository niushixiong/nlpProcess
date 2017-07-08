# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 18:11:46 2016

@author: shixiong
"""
import nltk
import numpy
import scipy
import nltk
import jieba
from pattern.en import suggest
import string
from config import Config as conf
from nltk.corpus import stopwords
import chardet
import pandas
import jieba.analyse
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer   
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
try:
    from textParserRead import textParserRead
except:
    from process.textParserRead import textParserRead
class textProcess:
    
    
    """初始化 
    """
    def __init__(self):
        self.data=0  
    def WordTokener(self,sent):#将单句字符串分割成词  
         
        wordsInStr = nltk.word_tokenize(sent)  
        return wordsInStr    
    def WordCheck(self,words):#拼写检查
       
        checkedWords=[]
        for word in words:
            WLis=suggest(word)
            if  len(WLis)>1:
                word=WLis[0][0]
            checkedWords.append(word)
        return checkedWords
    def CleanLines(self,line):
        identify =string.maketrans('', '')  
        delEStr =string.punctuation + string.digits #ASCII 标点符号，数字    
#         cleanLine= line.translate(identify, delEStr) #去掉ASCII 标点符号和空格  
        cleanLine =line.translate(identify,delEStr) #去掉ASCII 标点符号  
        return cleanLine
    def CleanWords(self,wordsInStr):#去掉标点符号，长度小于3的词以及non-alpha词，小写化
        cleanWords=[]
        stopwd = stopwords.words('english')
        cleanWords=[w.lower() for w in wordsInStr if w.lower() not in stopwd and 3<=len(w)]
        return cleanWords
    def StemWords(self,cleanWordsList):
        stemWords=[]
        for w in cleanWordsList:
            wp=wn.morphy(w)
            if wp!=None:
                if isinstance(wp, unicode):
                    stemWords.append(wp.encode("utf-8"))
                else:
                    stemWords.append(wp)
            else:
                if isinstance(w, unicode):
                    stemWords.append(w.encode("utf-8"))
                else:
                    stemWords.append(w)
        return stemWords

    def tokenProcessEn(self,rawdata,low_freq_filter = False,remove_stopwords=False):    
       
        #句子去标点数字分割单词
        cleanLines=self.CleanLines(rawdata)
        words=self.WordTokener(cleanLines)  
        #checkedWords=self.WordCheck(words)#暂不启用拼写检查  
        cleanWords=self.CleanWords(words)
       # print cleanWords
        stemWords=self.StemWords(cleanWords)  
      #  print stemWords
#            cleanWords=self.CleanWords(stemWords)#第二次清理出现问题，暂不启用  
       # strLine=self.WordsToStr(stemWords)  
        return stemWords
        
    """
        #简化的 中文+英文 预处理
        1.去掉停用词
        2.去掉标点符号
        3.处理为词干
        4.去掉低频词
        rawdata 为str类型
        return text ( a list)
    """     
    def tokenProcess(self,rawdata,low_freq_filter = False,remove_stopwords=False,english=False):
#       对于每个文档的提取前10个关键词
#        texts_tokenized = []
#        for document in rawdata:
#            texts_tokenized_tmp = []
#            for word in word_tokenize(document):
#                texts_tokenized_tmp += jieba.analyse.extract_tags(word,10)
#            texts_tokenized.append(texts_tokenized_tmp)   

        #texts-tokeized 为list 里面unicode   
   
        try:
            texts_tokenized=list(jieba.cut(rawdata))
        except:
            print rawdata
            print type(rawdata)
            print chardet.detect(rawdata)
            print rawdata.decode("utf-8")
            texts_tokenized=list(jieba.cut(rawdata.decode("utf-8")))
        #读取停顿词文件中的词，加入stopword中  
        if remove_stopwords:
            stopword=[]
            fp=open("G:\worksp\pythonScrapy\src\process\stopword.txt", "r")
            while 1:
                #line  str类型
                line=fp.readline()
                if not line:
                    break
                line=line.strip()
                #print "stopword："
                #print line
                stopword.append(line)
           
            fp.close()   
    #       
            #print "****************************************************"
            #去掉停顿词
            texts_filtered_stopwords=[]
            for i in texts_tokenized: 
                i=i.encode('utf-8')
                if i not in stopword:
                    texts_filtered_stopwords.append(i)
            #print "去掉停顿词后："
        else:
            texts_filtered_stopwords=[]
            for i in texts_tokenized: 
                i=i.encode('utf-8')
                
                texts_filtered_stopwords.append(i)
            
        
        #print "*****************************************************"
        #去除标点符号
        english_punctuations = ['，',',', '。','.', '：',';', '；',':', '？','?', '（', '）', '【', '】', '&', '！', '*', '@', '#', '￥', '%','、','/','(',')','[',']','!','$']
        texts_filtered = [word for word in texts_filtered_stopwords if word not in  english_punctuations]
        #词干化
        #print "****************************************************"
        if english:
            st = LancasterStemmer()
            texts_stemmed = [st.stem(word) for word in  texts_filtered]
            print "词干化"
        else:
            texts_stemmed=texts_filtered
        #去除过低频词
        if low_freq_filter:
            #print "****************************************************"
            stems_once = set(stem for stem in set(texts_stemmed) if texts_stemmed.count(stem) == 1)
            texts = [stem for stem in texts_stemmed if stem not in stems_once ]
        else:
            texts = texts_stemmed
        #print "****************************************************"
        return texts
        
if __name__=="__main__":
   # tpr=textParserRead()
    #tpr.getFramdata()
    t=textProcess()
    lis1="一种头枕颈枕靠背合一的座椅，可用于飞机、火车、长途客车、中巴车和剧院、办公室、家庭、公共场所，它是在座椅靠背前面上部和头枕体的前面下部增加一个颈枕体，将头枕体、颈枕体和座椅靠背三者合为一体，构成一个头枕、颈枕、座椅靠背组合体。头枕、颈枕、座椅靠背组合体一同随座椅靠背调整转动，对乘坐人的头部、颈部和背部同时提供支撑和保护，防止乘坐人颈部疲劳，当乘坐人的飞机、火车、长途客车、中巴车加速向前运动时，或发生被追尾碰撞的交通事故时，乘坐人最易受伤的颈部受到有效地支撑和保护，防止颈部受伤。"
    lis="At the end , when the now computerized Yoda finally reveals his martial artistry , the film ascends to a kinetic life so teeming that even cranky adults may rediscover the quivering kid inside ."
    res=t.tokenProcessEn(lis)
    res1=t.tokenProcess(lis1)
    