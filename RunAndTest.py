
# -*- coding: utf-8 -*-
'''Created on 2017年1月11日

@author: shixiong
'''

#!/usr/bin/env python
import pandas as pd

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import affinity_propagation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
try: 
    from Word2VecUtility import Word2VecUtility
except:
    from process.Word2VecUtility import Word2VecUtility
from sklearn import metrics  
try:
    from createModels import Createmodels
except:
    from process.createModels import Createmodels
try:
    from textParserRead import textParserRead
except:
    from process.textParserRead import textParserRead

import sentiment as sentClasstify

from numpy import vstack
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from astropy.nddata.utils import overlap_slices
from random import shuffle
import os
import shutil
def deleteTrainModel(rootdir="",filepath=None):
    filelist=[]
    
    if filepath!=None:
        os.remove(filepath)
        print filepath+" removed!"
    else:
        filelist=os.listdir(rootdir)
        for f in filelist:
            filepath = os.path.join( rootdir, f )
            if os.path.isfile(filepath):
                os.remove(filepath)
                print filepath+" removed!"
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath,True)
                print "dir "+filepath+" removed!"
            
def runRepeatDifParam():
    df=None
    try:
        if os.path.exists("G:/worksp/pythonScrapy/src/process/extract-stanfordSentimentTreebank-master/sst2_train_sentence.csv"):
            train=pd.read_csv("G:/worksp/pythonScrapy/src/process/extract-stanfordSentimentTreebank-master/sst2_train_sentence.csv")
            test=pd.read_csv("G:/worksp/pythonScrapy/src/process/extract-stanfordSentimentTreebank-master/sst2_test.csv")
        else:
            print "file not found"
    except:
        print "file read fail"

   
    createmodelBOW=Createmodels(df)
    createmodelBOWFeatureExct=Createmodels(df)
   # createmodelBOWCHM=Createmodels(df)
   # createmodelBOWCHV=Createmodels(df)
    #createmodelWAV=Createmodels(df)
   # createmodelWBC=Createmodels(df)
   ## createmodelBOWCluster=Createmodels(df)
   # createmodelDAE=Createmodels(df)
    
    #拼接模型
    #createmodelWLink=Createmodels(df1)
    #距离模型
    #createmodelWMD=Createmodels(df1)
    
    print "not feature extract:" 
    """bag of word model random forest svm classtify"""   
    #createmodelBOW.createBagofWord(tfidfset=True,dataIfSet=True,train=train,test=test, selectF=False,remove_stopwords=False)
   # testVec=createmodelBOW.testDataVec
    #trainVec=createmodelBOW.trainDataVec
    #bag of word randomForest  
    #sentClasstify.classtifyForest(trainVec,testVec,createmodelBOW.train,createmodelBOW.test,"BoW") 
    #bag of word svm     
    #sentClasstify.classtifySvm(trainVec,testVec,createmodelBOW.train,createmodelBOW.test,"BoW")
    
    """bag of word model and feature conbine random forest svm classtify""" 
    #createmodelBOWFeatureExct.createBOWUseNewFeatureExtract(tfidfset=True,dataIfSet=True,train=train,test=test,remove_stopwords=False ,selectF=False,low_freq_filter=False)
   # testVec=createmodelBOWFeatureExct.testDataVec
   # trainVec=createmodelBOWFeatureExct.trainDataVec
    #bag of word randomForest  
    #sentClasstify.classtifyForest(trainVec,testVec,createmodelBOWFeatureExct.train,createmodelBOWFeatureExct.test,"BoWFeatureExct") 
    #bag of word svm     
    #sentClasstify.classtifySvm(trainVec,testVec,createmodelBOWFeatureExct.train,createmodelBOWFeatureExct.test,"BoWFeatureExct")
    #deleteTrainModel(filepath="G:/worksp/pythonScrapy/src/process/model/500features_5minwords_10context")
   
    print "feature Extract:"
    for param in range(6,10,1):
        simlar=float(param)/10
        FeatureNums=[5000,6000,7000,8000,9000,10000,11000,12000]
        for featureNum in FeatureNums:
            f=open("./sst/FeatureConbineOrNotprecison.txt",'a')
            f.write("\n精度:\n")
            f.write("similar:"+str(simlar)+"\n")
            f.close()     
            #bag of word model random forest svm classtify  
            createmodelBOW.createBagofWord(tfidfset=True,dataIfSet=True,train=train,test=test, selectF=True,fs_method='IG',fs_num=featureNum,remove_stopwords=False)
            testVec=createmodelBOW.testDataVec
            trainVec=createmodelBOW.trainDataVec
            #bag of word randomForest  
            sentClasstify.classtifyForest(trainVec,testVec,createmodelBOW.train,createmodelBOW.test,"BoW") 
            #bag of word svm     
            sentClasstify.classtifySvm(trainVec,testVec,createmodelBOW.train,createmodelBOW.test,"BoW")
            
            #bag of word model and feature conbine random forest svm classtify 
            createmodelBOWFeatureExct.createBOWUseNewFeatureExtract(tfidfset=True,dataIfSet=True,train=train,test=test,remove_stopwords=False ,selectF=True,low_freq_filter=False,fs_method='IG', fs_num=featureNum,similarVal=simlar)
            testVec=createmodelBOWFeatureExct.testDataVec
            trainVec=createmodelBOWFeatureExct.trainDataVec
            #bag of word randomForest  
            sentClasstify.classtifyForest(trainVec,testVec,createmodelBOWFeatureExct.train,createmodelBOWFeatureExct.test,"BoWFeatureExct") 
            #bag of word svm     
            sentClasstify.classtifySvm(trainVec,testVec,createmodelBOWFeatureExct.train,createmodelBOWFeatureExct.test,"BoWFeatureExct")
            deleteTrainModel(filepath="G:/worksp/pythonScrapy/src/process/model/500features_5minwords_10context")
    
    
    
    """ 测试拼接模型"""
  #  overlappingRate=0.1
   # createmodelWLink.WordsLink(overlappingRate)
    #testVecLink=createmodelWLink.testDataVec
    #trainVecLink=createmodelWLink.trainDataVec
    #classtify(trainVecLink,testVecLink,createmodelWLink.train,createmodelWLink.test,1) 
    
if __name__=="__main__":
    runRepeatDifParam()
  