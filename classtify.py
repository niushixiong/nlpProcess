# -*- coding: utf-8 -*-
"""
Created on 6.2  14:10:29 2016

@author: shixiong
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
    from process.Word2VecUtility import Word2VecUtility
    from process.createModels import Createmodels
    from process.textParserRead import textParserRead
except:
    from Word2VecUtility import Word2VecUtility
    from createModels import Createmodels
    from textParserRead import textParserRead
from sklearn import metrics  

from numpy import vstack
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from astropy.nddata.utils import overlap_slices
from random import shuffle



"""
#处理dateframe['IPCNum']，里面含有多项ipcnum ,根据搜索项'A61C1/08 and 手机','A61B5/0245 and 手机',
 'A61B5/11 and 手机','H04M1/12 and 手机','H04N7/18 and 手机','A63F13/24 and 手机','A63F13/90 and 手机',
 'H04M1/02 and 手机',如果IPCNum中有以上搜索项中IPCnum，则取ipcNUM，如果没有则取第一项
 ['手机 and 处理器',#'手机  and 显示器',#'手机  and 摄像头',#'手机 and 软件',#'手机 and 电池']
"""
def processIPCNum(dataframe):
    IPCNums=[]
    #existIPCNums=['A61C1/08','A61B5/0245','A61B5/11','H04M1/12','H04N7/18','A63F13/24','A63F13/90','H04M1/02']
    existIPCNums=['A47','B23','B25','B64','B60','B66','F01','F16','G08','A61']
    
    counter=0
    for rawIPCNum in dataframe['IPCNum']:
        IPCNums=rawIPCNum.strip().split(';')
        flag=False
        for i in range(len(IPCNums)):
            if existIPCNums.__contains__(IPCNums[i].strip()[:3]):
                dataframe['IPCNum'][counter]=IPCNums[i].strip()
                flag=True
                break
        #if not flag:
            #existIPCNums.append(IPCNums[0].strip()[:3])
            #dataframe['IPCNum'][counter]=IPCNums[0].strip()
        counter=counter+1
    return dataframe
def IPCNumFileter(dataframe,num):
    #'A47' 299,'B23' 560,'B25' 315,'B64' 7638,'B60' 1362,'B66' 210,'F01' 310,'F16' 1090,'G08' 644
   # existIPCNums=['A47','B23','B25','B64','B60','B66','F01','F16','G08']
    #existIPCNums=['B23','B64','B60','F16','G08']
    #existIPCNums=['B64F5','B64F1','B64C1','B64C2']
    existIPCNums=['B64','A61F1','A47','F16']
    #counts=[0,0,0,0,0,0,0,0,0]
    counts=[0,0,0,0]
    lis=[]
    lists = [[] for i in range(len(counts))]
    i=0
    for IPCNum in dataframe['IPCNum']:
        try:
            k=existIPCNums.index(IPCNum.strip()[:3])
            if(k>=0 and k<=len(counts)-1):
                lists[k].append(i)
                counts[k]+=1
            """    
            if(k>=0 and k<=8 and counts[k]<num):
            
                lis.append(i)
                counts[k]+=1
            """
        except :
            i+=1
            continue
        
        i+=1
    #随机从lists 取出Num*9个
    for li in lists:
        Len=len(li)
        randLis=range(Len)
        shuffle(randLis)
        lis.extend(li[:num])
    # 改变索引值为1-N
    shuffle(lis)
    resData=dataframe.ix[lis]
    resData.index=range(len(lis))
    return resData    
def calculate_accurate(actual,predict,classes):
    m_precision = metrics.accuracy_score(actual,predict)
    print '结果计算:'
    print '精度:{0:.3f}'.format(m_precision)
    f=open("./precison.txt",'a')
    f.write(classes)
    f.write('精度:{0:.3f}'.format(m_precision))
    f.write('\n')
    f.close()
 
    
def classtifyForest(trainDataVecs,testDataVecs,train,test,flag):
    
    # ****** Fit a random forest and extract predictions
    #
    forest = RandomForestClassifier()#n_estimators = 100)

    # Fitting the forest may take a few minutes
    print "Fitting a random forest to labeled training data..."
    temp=train['IPCNum'].get_values()
    trainLabel=[]
    for t in temp:
        trainLabel.append(t[:3])
    temp=test['IPCNum'].get_values()
    print "train label num: ",len(trainLabel)
    testLabel=[]
    for t in temp:
        testLabel.append(t[:3])
    print "test label num: ",len(testLabel)
    print "trainVec shape:",trainDataVecs.shape
    
    forest = forest.fit(trainDataVecs,trainLabel)#取分类号的前三位
    print "testVec shape:",testDataVecs.shape
    result = forest.predict(testDataVecs)
    # 添加计算准确度
    actual=testLabel# 取分类号的前三位
    
    actual=np.array(actual)
    #print actual
    #print predicted
    # 计算分类各种参数指标
    calculate_accurate(actual,result,"randomForest")
    # 计算完毕
    # Write the test results
    output = pd.DataFrame(data={"id":test["idNum"],'title':test['title'], "IPCNum":result})
    if flag==1:
        output.to_csv("result1.csv", index=False, quoting=None)
        print "Wrote result1.csv"
    elif flag==2:
        output.to_csv("result2.csv", index=False, quoting=None)
        print "Wrote result2.csv"
    else: 
        output.to_csv("result3.csv", index=False, quoting=None)
        print "Wrote result3.csv"
"""  param：datavecs训练的矩阵，dataTitle 训练数据的标题"""      
def Clusterpredict(dataVecs,dataTitle,dataNum):
    print 'Start Kmeans:'
    # kmeans 训练
    clf = KMeans(n_clusters=10)
    DataLabels = clf.fit_predict(dataVecs)
    print "kmeans finally"
    
    print DataLabels
    # 将dataTitle 和dataNum两个list合并
    for i in range(len(dataTitle)):
        dataTitle[i]="title:"+dataTitle[i]+" ipcNum:"+dataNum[i]
    sample_label_map = dict(zip(dataTitle, DataLabels ))
    # Print the ten clusters
    for cluster in xrange(0,10):
        # Print the cluster number
        print "\nCluster %d" % cluster
        # 结果写入文件
        fl=open('kmeans_result'+str(cluster)+'.txt', 'w')
        #
        # Find all of the samples datas for that cluster number, and 写入文件
        for i in xrange(0,len(sample_label_map.values())):
            if( sample_label_map.values()[i] == cluster ):
                fl.write(sample_label_map.keys()[i])
                fl.write(" "+str(cluster))
                fl.write("\n")
        fl.close()
    #10个中心点
    print "10个中心点:"
    #f2=open('vector.txt', 'w')
    #f2.write(clf.cluster_centers_)
    np.savetxt("vector.txt",clf.cluster_centers_,fmt='%s',newline='\n')
    #print(clf.cluster_centers_)
    
    #每个样本所属的簇
    print "每个样本所属的簇:"
    print(clf.labels_)
    i = 1
    while i <= len(clf.labels_):
        print i, clf.labels_[i-1]
        i = i + 1

    #用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print(clf.inertia_)
""" KNN  use WMD model , trainDatavecs:any array of any form,testdataDistancematrics:np.array train:dataframe;test:dataframe
    this special metric is precomputed ,so the tarindatavecs can be some forms(), because the distanse is defined
"""  
def classtifyKnnWmd(trainDataVecs,testDataDistanceMatrics,train,test):
    neigh = KNeighborsClassifier()
    temp=train['IPCNum'].get_values()
    trainLabel=[]
    for t in temp:
        trainLabel.append(t[:3])
    temp=test['IPCNum'].get_values()
    testLabel=[]
    for t in temp:
        testLabel.append(t[:3])
    neigh.metric = 'precomputed'
    neigh.fit(trainDataVecs, trainLabel) 
    predictData=neigh.predict(testDataDistanceMatrics)
    actual=testLabel
    actual=np.array(actual)
    #print actual
    #print predicted
    # 计算分类各种参数指标
    calculate_accurate(actual,predictData,"LinearSvm")
    # 计算完毕
    # Write the test results
    output = pd.DataFrame(data={"id":test["idNum"],'title':test['title'], "IPCNum":predictData})
    output.to_csv("resultKNNWMD.csv", index=False, quoting=3)
    print "Wrote resultKNNWMD.csv"
def classtifySvm(trainDataVecs,testDataVecs,train,test,flag):
    #应用linear_svm算法 输入词袋向量和分类标签
    #svclf = SVC(kernel = 'linear')   # default with 'rbf'
    svclf = LinearSVC(penalty="l1",dual=False, tol=1e-4)
    temp=train['IPCNum'].get_values()
    trainLabel=[]
    for t in temp:
        trainLabel.append(t[:3])
    temp=test['IPCNum'].get_values()
    testLabel=[]
    for t in temp:
        testLabel.append(t[:3])
    
    svclf.fit(trainDataVecs, trainLabel)
    # 预测分类结果
    predicted = svclf.predict(testDataVecs)
    actual=testLabel
    
    actual=np.array(actual)
    #print actual
    #print predicted
    # 计算分类各种参数指标
    calculate_accurate(actual,predicted,"LinearSvm")
    # 计算完毕
    # Write the test results
    output = pd.DataFrame(data={"id":test["idNum"],'title':test['title'], "IPCNum":predicted})
    if flag==1:
        output.to_csv("resultsvm1.csv", index=False, quoting=None)
        print "Wrote resultsvm1.csv"
    elif flag==2:
        output.to_csv("resultsvm2.csv", index=False, quoting=None)
        print "Wrote resultsvm2.csv"
    else: 
        output.to_csv("resultsvm3.csv", index=False, quoting=None)
        print "Wrote resultsvm3.csv"
        
def classtifyLogisticRegress(trainDataVecs,testDataVecs,train,test,flag):
    
 
    print "Fitting logisticRegression to labeled training data..."
   
    clf = LogisticRegression(class_weight="auto")
    
    clf.fit(testDataVecs, train["IPCNum"][:3])
    result = clf.predict_proba(testDataVecs)
    # 添加计算准确度
    actual=test['IPCNum'][:3]
    
    actual=np.array(actual)
    #print actual
    #print predicted
    # 计算分类各种参数指标
    calculate_accurate(actual,result,'logisticRegression')
    #计算完毕
    # Write the test results
    output = pd.DataFrame(data={"id":test["idNum"],'title':test['title'], "IPCNum":result})
    if flag==1:
        output.to_csv("LogisticRegressresult1.csv", index=False, quoting=None)
        print "Wrote result1.csv"
    elif flag==2:
        output.to_csv("LogisticRegressresult2.csv", index=False, quoting=None)
        print "Wrote result2.csv"
    else: 
        output.to_csv("LogisticRegressresult3.csv", index=False, quoting=None)
        print "Wrote result3.csv"
    
if __name__=="__main__":
   
    df=None
    if os.path.exists("./data/data_process.csv"):
        df=pd.read_csv("./data/data_process.csv")
    elif os.path.exists("./data/data.csv"):
        df  = pd.read_csv('./data/data.csv')
        for i in range(len(df["description"])):
            
            df["description"][i]=df["description"][i].decode("gbk").encode("utf-8")
            df["title"][i]=df["title"][i].decode("gbk").encode("utf-8")
            df["IPCNum"][i]=df["IPCNum"][i].decode("gbk").encode("utf-8")
            df["idNum"][i]=df["idNum"][i].decode("gbk").encode("utf-8")
        #    if i>9850:
                
         #       print str(i),":",df["description"][i]
            if i%1000==0:
                print df["description"][i]
        print "dateframe description读取成功"
        df.to_csv("./data/data_process.csv",columns={'idNum','title','IPCNum','description'})
    else:
        tpr=textParserRead()
        #得到dataframe 类型的原始数据
        df=tpr.getFramdata()
        print "start"
        print "processIPCNum;"
        dateframe=processIPCNum(df)
        dateframe.to_csv("./data/data.csv",columns={'idNum','title','IPCNum','description'})
       # df.to_csv("./data/data.csv")
        print " process ipcnum end"
    #df1=df.ix[:951]
   
    df1=IPCNumFileter(df,280)
   
    createmodelBOW=Createmodels(df1)
    createmodelWAV=Createmodels(df1)
    createmodelWBC=Createmodels(df1)
   # createmodelBOWCluster=Createmodels(df)
    createmodelDAE=Createmodels(df1)
    #拼接模型
    #createmodelWLink=Createmodels(df1)
    #距离模型
    #createmodelWMD=Createmodels(df1)
    
    f=open("./precison.txt",'a')
    f.write("\n精度:\n")
    f.close()  
    
    
    """ 测试拼接模型"""
  #  overlappingRate=0.1
   # createmodelWLink.WordsLink(overlappingRate)
    #testVecLink=createmodelWLink.testDataVec
    #trainVecLink=createmodelWLink.trainDataVec
    #classtify(trainVecLink,testVecLink,createmodelWLink.train,createmodelWLink.test,1) 
    
    

    
    """bag of word random forest pass"""   
    createmodelBOW.createBagofWord(True)
    testVec=createmodelBOW.testDataVec
    trainVec=createmodelBOW.trainDataVec
  
    classtifyForest(trainVec,testVec,createmodelBOW.train,createmodelBOW.test,1) 
    """bag of word svm """    
    classtifySvm(trainVec,testVec,createmodelBOW.train,createmodelBOW.test,1)
    
    """  autoEncoder matrix model"""
    print "start autoEncoder"
   
    if not os.path.exists("./DAEtestVec2.bin"):
        createmodelDAE.CreateAutoEncodeModel(False)
        DAEtestVec=createmodelDAE.testDataVec
        DAEtrainVec=createmodelDAE.trainDataVec
        DAEtestVec.tofile("./DAEtestVec2.bin")
        DAEtrainVec.tofile("./DAEtrainVec2.bin")
        createmodelDAE.train.to_csv("./data/train.csv",columns={'idNum','title','IPCNum','description'})
        createmodelDAE.test.to_csv("./data/test.csv",columns={'idNum','title','IPCNum','description'})
        DAEtestVec=DAEtestVec.reshape(DAEtestVec.shape[0],DAEtestVec.shape[1]*DAEtestVec.shape[2])
        DAEtrainVec=DAEtrainVec.reshape(DAEtrainVec.shape[0],DAEtrainVec.shape[1]*DAEtrainVec.shape[2])
    else:
        DAEtestVec=np.fromfile("./DAEtestVec2.bin", dtype=np.float)
        DAEtrainVec=np.fromfile("./DAEtrainVec2.bin", dtype=np.float) # 按照float类型读入数据
        createmodelDAE.train=pd.read_csv('./data/train.csv')
        createmodelDAE.test=pd.read_csv('./data/test.csv')
        print "shape:test,train:" ,DAEtestVec.shape,DAEtrainVec.shape
        print len(createmodelDAE.test),DAEtestVec.shape[0]/len(createmodelDAE.test)
        DAEtestVec.shape=len(createmodelDAE.test),DAEtestVec.shape[0]/len(createmodelDAE.test)
        DAEtrainVec.shape=len(createmodelDAE.train),DAEtrainVec.shape[0]/len(createmodelDAE.train)
        
    print "encoder classfity"
    classtifyForest(DAEtrainVec,DAEtestVec,createmodelDAE.train,createmodelDAE.test,2)  
    """autoEncoder matrix model svm """   
    classtifySvm(DAEtrainVec,DAEtestVec,createmodelDAE.train,createmodelDAE.test,2)
    print "autoEncoder end"
   
    """" word2vec averageVector random forest pass"""   
    createmodelWAV.createWord2vec(False)
    WAVtestVec=createmodelWAV.testDataVec
    WAVtrainVec=createmodelWAV.trainDataVec
    classtifyForest(WAVtrainVec,WAVtestVec,createmodelWAV.train,createmodelWAV.test,2)  
    """word2vec averageVector svm """   
    classtifySvm(WAVtrainVec,WAVtestVec,createmodelWAV.train,createmodelWAV.test,2)
   
    
    """" word2vec bagofcentoid random forest pass"""     
    createmodelWBC.getWord2vecBagofCenter()
    WBCtestVec=createmodelWBC.testDataVec
    WBCtrainVec=createmodelWBC.trainDataVec
    classtifyForest(WBCtrainVec,WBCtestVec,createmodelWBC.train,createmodelWBC.test,3)  
    """word2vec bagofcentoid svm """   
    classtifySvm(WBCtrainVec,WBCtestVec,createmodelWBC.train,createmodelWBC.test,3)
    
    
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
