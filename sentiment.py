# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:24:01 2016

@author: sx
"""

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
from numpy import vstack
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from astropy.nddata.utils import overlap_slices
from random import shuffle


def calculate_accurate(actual,predict,classes):
    m_precision = metrics.accuracy_score(actual,predict)
    print '结果计算:'
    print '精度:{0:.3f}'.format(m_precision)
    f=open("./sst/doc2vecModelFeature.txt",'a')
    f.write(classes)
    f.write('精度:{0:.3f}'.format(m_precision))
    f.write('\n')
    f.close()
 
    
def classtifyForest(trainDataVecs,testDataVecs,train,test,resultName):
    
    # ****** Fit a random forest and extract predictions
    #
    forest = RandomForestClassifier()#n_estimators = 100)

    # Fitting the forest may take a few minutes
    print "Fitting a random forest to labeled training data..."
    trainLabel=train['label']
    testLabel=test['label']

    forest = forest.fit(trainDataVecs,trainLabel)#取分类号的前三位

    result = forest.predict(testDataVecs)
    # 添加计算准确度
    actual=testLabel# 取分类号的前三位
    
    actual=np.array(actual)
    #print actual
    #print predicted
    # 计算分类各种参数指标
    calculate_accurate(actual,result,resultName+"RandomForest")
    # 计算完毕
    # Write the test results
    output = pd.DataFrame(data={"content":test["content"], "label":result})
   
    output.to_csv("G:\\worksp\\pythonScrapy\\src\\process\\extract-stanfordSentimentTreebank-master\\result\\resultRandomForest_"+resultName+".csv", index=False, quoting=None)
    print "Wrote resultRandomForestResult_"+resultName+".csv"

  
""" KNN  use WMD model , trainDatavecs:any array of any form,testdataDistancematrics:np.array train:dataframe;test:dataframe
    this special metric is precomputed ,so the tarindatavecs can be some forms(), because the distanse is defined
"""  
def classtifyKnnWmd(trainDataVecs,testDataDistanceMatrics,train,test):
    neigh = KNeighborsClassifier()
   
    trainLabel=train['label']
   
    testLabel=test['label']
    
    neigh.metric = 'precomputed'
    neigh.fit(trainDataVecs, trainLabel) 
    predictData=neigh.predict(testDataDistanceMatrics)
    actual=testLabel
    actual=np.array(actual)
    #print actual
    #print predicted
    # 计算分类各种参数指标
    calculate_accurate(actual,predictData,"KnnWmd")
    # 计算完毕
    # Write the test results
    output = pd.DataFrame(data={"content":test["content"], "label":predictData})
    output.to_csv("resultKNNWMD.csv", index=False, quoting=2)
    print "Wrote resultKNNWMD.csv"
def classtifySvm(trainDataVecs,testDataVecs,train,test,resultName):
    #应用linear_svm算法 输入词袋向量和分类标签
    #svclf = SVC(kernel = 'linear')   # default with 'rbf'
    svclf = LinearSVC(penalty="l1",dual=False, tol=1e-4)
  
    trainLabel=train['label']
   
   
    testLabel=test['label']
    
    
    svclf.fit(trainDataVecs, trainLabel)
    # 预测分类结果
    predicted = svclf.predict(testDataVecs)
    actual=testLabel
    
    actual=np.array(actual)
    #print actual
    #print predicted
    # 计算分类各种参数指标
    calculate_accurate(actual,predicted,resultName+"LinearSvm")
    # 计算完毕
    # Write the test results
    output = pd.DataFrame(data={"content":test["content"],'label':predicted})
    
    output.to_csv("G:\\worksp\\pythonScrapy\\src\\process\\extract-stanfordSentimentTreebank-master\\result\\resultsvm_"+resultName+".csv", index=False, quoting=None)
    print "Wrote resultsvm_"+resultName+".csv"
   
        
def classtifyLogisticRegress(trainDataVecs,testDataVecs,train,test,resultName):
    
 
    print "Fitting logisticRegression to labeled training data..."
   
    clf = LogisticRegression(class_weight="auto")
    
    clf.fit(testDataVecs, train["label"])
    result = clf.predict_proba(testDataVecs)
    # 添加计算准确度
    actual=test['label']
    
    actual=np.array(actual)
    #print actual
    #print predicted
    # 计算分类各种参数指标
    calculate_accurate(actual,result,resultName+'logisticRegression')
    #计算完毕
    # Write the test results
    output = pd.DataFrame(data={"content":test["content"],"label":result})
   
       
    output.to_csv("G:\\worksp\\pythonScrapy\\src\\process\\extract-stanfordSentimentTreebank-master\\result\\LogisticRegressresult_"+resultName+".csv", index=False, quoting=None)
    print "Wrote resultLogisticRegressreslut_"+resultName+".csv"
    
if __name__=="__main__":
    df=None
    try:
        if os.path.exists("G:/worksp/pythonScrapy/src/process/extract-stanfordSentimentTreebank-master/sst2_train_sentence.csv"):
            train=pd.read_csv("G:/worksp/pythonScrapy/src/process/extract-stanfordSentimentTreebank-master/sst2_train_sentence.csv")
            test=pd.read_csv("G:/worksp/pythonScrapy/src/process/extract-stanfordSentimentTreebank-master/sst2_test.csv")
        else:
            print "file not found"
    except:
        print "file read fail"
    createmodelDoc=Createmodels(df)
    createmodelDoc.createDoc2vec(dataIfSet=True, train=train, test=test, remove_stopwords=False, low_freq_filter=False,dm=0)    
    f=open("./sst/doc2vecModelFeature.txt",'a')
    f.write("\n精度:\n")
    f.close()  
    testVec=createmodelDoc.testDataVec
    trainVec=createmodelDoc.trainDataVec
    #bag of word randomForest  
    classtifyForest(trainVec,testVec,createmodelDoc.train,createmodelDoc.test,"Doc") 
    #bag of word svm     
    classtifySvm(trainVec,testVec,createmodelDoc.train,createmodelDoc.test,"Doc")
else:
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
    
    f=open("./sst/FeatureConbineOrNotprecison.txt",'a')
    f.write("\n精度:\n")
    f.close()  
    
    
    """ 测试拼接模型"""
  #  overlappingRate=0.1
   # createmodelWLink.WordsLink(overlappingRate)
    #testVecLink=createmodelWLink.testDataVec
    #trainVecLink=createmodelWLink.trainDataVec
    #classtify(trainVecLink,testVecLink,createmodelWLink.train,createmodelWLink.test,1) 
    
    

    
    """bag of word model random forest svm classtify"""   
    createmodelBOW.createBagofWord(tfidfset=True,dataIfSet=True,train=train,test=test, selectF=False,fs_method='IG',fs_num=11000,remove_stopwords=False)
    testVec=createmodelBOW.testDataVec
    trainVec=createmodelBOW.trainDataVec
    #bag of word randomForest  
    classtifyForest(trainVec,testVec,createmodelBOW.train,createmodelBOW.test,"BoW") 
    #bag of word svm     
    classtifySvm(trainVec,testVec,createmodelBOW.train,createmodelBOW.test,"BoW")
    
    """bag of word model and feature conbine random forest svm classtify""" 
    createmodelBOWFeatureExct.createBOWUseNewFeatureExtract(tfidfset=True,dataIfSet=True,train=train,test=test,remove_stopwords=False ,selectF=False,low_freq_filter=False,fs_method='IG', fs_num=11000)
    testVec=createmodelBOWFeatureExct.testDataVec
    trainVec=createmodelBOWFeatureExct.trainDataVec
    #bag of word randomForest  
    classtifyForest(trainVec,testVec,createmodelBOWFeatureExct.train,createmodelBOWFeatureExct.test,"BoWFeatureExct") 
    #bag of word svm     
    classtifySvm(trainVec,testVec,createmodelBOWFeatureExct.train,createmodelBOWFeatureExct.test,"BoWFeatureExct")
#    
#    
#    """bag of word combine word2vec random forest svm classtify max simlar"""   
#    createmodelBOWCHM.createBagofWordConW2V(googlenews=False,tfidfset=True,dataIfSet=True,train=train,test=test,selectF=False,selectMax=True,remove_stopwords=False)
#    testVec=createmodelBOWCHM.testDataVec
#    trainVec=createmodelBOWCHM.trainDataVec
#    #bag of word  conbine word2vec randomForest  
#    classtifyForest(trainVec,testVec,createmodelBOWCHM.train,createmodelBOWCHM.test,"BoWCHM") 
#    #bag of word combine word2vec svm     
#    classtifySvm(trainVec,testVec,createmodelBOWCHM.train,createmodelBOWCHM.test,"BoWCHM")
#    
#    """bag of word combine word2vec random forest svm classtify avg simlar"""   
#    createmodelBOWCHV.createBagofWordConW2V(googlenews=False,tfidfset=True,dataIfSet=True,train=train,test=test,selectF=False,selectMax=False,remove_stopwords=False)
#    classtifyForest(createmodelBOWCHV.trainDataVec,createmodelBOWCHV.testDataVec,createmodelBOWCHV.train,createmodelBOWCHV.test,"BoWCHV") 
#    #bag of word combine word2vec svm     
#    classtifySvm(createmodelBOWCHV.trainDataVec,createmodelBOWCHV.testDataVec,createmodelBOWCHV.train,createmodelBOWCHV.test,"BoWCHV")
#    
#    
#    """" word2vec averageVector random forest pass"""   
#    createmodelWAV.createWord2vec(googlenews=False,dataIfSet=True,train=train,test=test,remove_stopwords=True)
#    WAVtestVec=createmodelWAV.testDataVec
#    WAVtrainVec=createmodelWAV.trainDataVec
#    classtifyForest(WAVtrainVec,WAVtestVec,createmodelWAV.train,createmodelWAV.test,"WAV")  
#    classtifySvm(WAVtrainVec,WAVtestVec,createmodelWAV.train,createmodelWAV.test,"WAV")
#   
#    
#    """" word2vec bagofcentoid random forest pass"""     
#    createmodelWBC.getWord2vecBagofCenter(dataIfSet=True,train=train,test=test,remove_stopwords=True)
#    WBCtestVec=createmodelWBC.testDataVec
#    WBCtrainVec=createmodelWBC.trainDataVec
#    classtifyForest(WBCtrainVec,WBCtestVec,createmodelWBC.train,createmodelWBC.test,"WBC")   
#    classtifySvm(WBCtrainVec,WBCtestVec,createmodelWBC.train,createmodelWBC.test,"WBC")
#        
#    
#    
#    
#    """  autoEncoder matrix model"""
#    print "start autoEncoder"
#   
#    if not os.path.exists("./DAEtestVec2.bin"):
#        createmodelDAE.CreateAutoEncodeModel(False,dataIfSet=True, train=train, test=test, remove_stopwords=False, low_freq_filter=False)
#        DAEtestVec=createmodelDAE.testDataVec
#        DAEtrainVec=createmodelDAE.trainDataVec
#        DAEtestVec.tofile("./DAEtestVec2.bin")
#        DAEtrainVec.tofile("./DAEtrainVec2.bin")
#        if createmodelDAE.train.get("description",None)!=None:
#            createmodelDAE.train.to_csv("./data/train.csv",columns={'idNum','title','IPCNum','description'})
#            createmodelDAE.test.to_csv("./data/test.csv",columns={'idNum','title','IPCNum','description'})
#        else:
#            createmodelDAE.train.to_csv("./data/train.csv",columns={'label','description'})
#            createmodelDAE.test.to_csv("./data/test.csv",columns={'label','description'})
#        DAEtestVec=DAEtestVec.reshape(DAEtestVec.shape[0],DAEtestVec.shape[1]*DAEtestVec.shape[2])
#        DAEtrainVec=DAEtrainVec.reshape(DAEtrainVec.shape[0],DAEtrainVec.shape[1]*DAEtrainVec.shape[2])
#    else:
#        DAEtestVec=np.fromfile("./DAEtestVec2.bin", dtype=np.float)
#        DAEtrainVec=np.fromfile("./DAEtrainVec2.bin", dtype=np.float) # 按照float类型读入数据
#        createmodelDAE.train=pd.read_csv('./data/train.csv')
#        createmodelDAE.test=pd.read_csv('./data/test.csv')
#        print "shape:test,train:" ,DAEtestVec.shape,DAEtrainVec.shape
#        print len(createmodelDAE.test),DAEtestVec.shape[0]/len(createmodelDAE.test)
#        DAEtestVec.shape=len(createmodelDAE.test),DAEtestVec.shape[0]/len(createmodelDAE.test)
#        DAEtrainVec.shape=len(createmodelDAE.train),DAEtrainVec.shape[0]/len(createmodelDAE.train)
#        
#    print "encoder classfity"
#    classtifyForest(DAEtrainVec,DAEtestVec,createmodelDAE.train,createmodelDAE.test,"DAE")  
#    """autoEncoder matrix model svm """   
#    classtifySvm(DAEtrainVec,DAEtestVec,createmodelDAE.train,createmodelDAE.test,"DAE")
#    print "autoEncoder end"
#   

    
    """bag of word kmeans""" 
    """
    createmodelBOWCluster.createBagofWord(True)
    testVec=createmodelBOWCluster.testDataVec
    trainVec=createmodelBOWCluster.trainDataVec
    
    DataVec=vstack((trainVec,testVec))
    datatitle=[]
    datatitle.extend(createmodelBOWCluster.train['title'])
    datatitle.extend(createmodelBOWCluster.test['title'])
    datanum=[]
    datanum.extend(createmodelBOWCluster.train['IPCNum'])
    datanum.extend(createmodelBOWCluster.test['IPCNum'])
    Clusterpredict(DataVec,datatitle,datanum)
    #classtify(trainVec,testVec,train,test,1)
    
    """
    ""
    
    """ 测试WMD模型"""
    """
    traindatavec=createmodelWMD.trainDataVec
    train=createmodelWMD.train
    test=createmodelWMD.test
    distanceMatrics=createmodelWMD.createWMD()
    classtifyKnnWmd(traindatavec, distanceMatrics, train, test)
    
    """
