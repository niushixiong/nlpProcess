# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 15:36:10 2016

@author: shixiong
"""
from gensim.models import Word2Vec
from gensim.models import doc2vec
import math

try:
    from FeatureExtraction  import FeatureExtractSimilarCon
except:
    from process.FeatureExtraction  import FeatureExtractSimilarCon

try:
    import textProcess
except:
    import process.textProcess
import os
try: 
    from Word2VecUtility import Word2VecUtility
except:
    from process.Word2VecUtility import Word2VecUtility
# import kaggle-word2vec-movie-reviews-master #导入不同包中的模块出错
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans
import time
import logging
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
try:
    from process import Doc2VecUtility_correct
except:
    import Doc2VecUtility_correct
from sklearn.ensemble import RandomForestClassifier
import numpy as np
try:
    from process.textParserRead import textParserRead
except:
    from textParserRead import textParserRead   
from numpy import array
import pandas as pd
try:
    from process.FeatureSelectionUtil import *
except:
    from FeatureSelectionUtil import * 



class Createmodels():
    """ data:dataframe 从textParserRead.getFramedata返回得到  """
    def __init__(self, data, feature=500):
        self.model = 0
        self.data = data  # dataframe 原始数据
        self.train = 0  # train dataframe
        self.test = 0
        self.trainDataVec = 0  # 训练数据的向量形式
        self.testDataVec = 0
        self.feature = feature  # 特征维数
        
        
    def Word2vecModel(self, googlenews=False, dataIfSet=False, train=None, test=None, remove_stopwords=False , low_freq_filter=False, model_name=0):
        
        """  从本地读取csv 文件建立 train test unsup集
        train  = pd.read_csv('../data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
        #test = pd.read_csv('../data/testData.tsv', header=0, delimiter="\t", quoting=3)
        unsup = pd.read_csv('../data/unlabeledTrainData.tsv', header=0,  delimiter="\t", quoting=3 )
        """
         
        if dataIfSet:
            self.train = train
            self.test = test
        else:
            df = self.data
            train, test = Word2VecUtility.split_train_test(df, test_portion=0.3)
            # list [[ str]]
            self.train = train
            self.test = test
        clean_train_reviews = []
        clean_test_reviews = []
        n_dim = self.feature   
        num_features = n_dim  # Word vector dimensionality
        min_word_count = 5  # Minimum word count
        num_workers = 4  # Number of threads to run in parallel
        context = 10  # Context window size
        downsampling = 1e-3
        if model_name == 0:
            # Downsample setting for frequent words
         
            model_name = "./model/%dfeatures_%dminwords_%dcontext" % (n_dim, min_word_count, context)
        if not os.path.exists(model_name): 
            sentences = []
            print "Parsing sentences from training set"
            colnames = train.columns.tolist()
            if colnames.__contains__("description") == True:
            # if train.get("description",None)!=None:
           
                for review in train["description"]:
                    temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_train_reviews.append(" ".join(sum(temp, [])))  # temp=[[1,2],[3,4]] sum(temp,[])=[1, 2, 3, 4]
        
                    # doc_terms_list_train.append(sum(temp,[]))
                print "Parsing sentences from unlabeled set"
                for review in test["description"]:
                    temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_test_reviews.append(" ".join(sum(temp, [])))
            else:

                for review in train["content"]:
                    temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_train_reviews.append(" ".join(sum(temp, [])))
                print "Parsing sentences from unlabeled set"
                for review in test["content"]:
                    temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_test_reviews.append(" ".join(sum(temp, [])))
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                                level=logging.INFO)

            num_features = n_dim  # Word vector dimensionality
            min_word_count = 5  # Minimum word count
            num_workers = 4  # Number of threads to run in parallel
            context = 10  # Context window size
            downsampling = 1e-3  # Downsample setting for frequent words
     
            print "Training Word2Vec model..."
            sentencesRepeat = sentences * 20
            self.model = Word2Vec(sentencesRepeat, workers=num_workers, alpha=0.025, sg=1, \
                            size=num_features, min_count=min_word_count, \
                            window=context, sample=downsampling, seed=1)
            self.model.init_sims(replace=True)
            self.model.save(model_name)
        else:
            sentences = []
            print "Parsing sentences from training set"
            colnames = train.columns.tolist()
            if colnames.__contains__("description") == True:
            # if train.get("description",None)!=None:
           
                for review in train["description"]:
                    temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_train_reviews.append(" ".join(sum(temp, [])))  # temp=[[1,2],[3,4]] sum(temp,[])=[1, 2, 3, 4]
        
                    # doc_terms_list_train.append(sum(temp,[]))
                print "Parsing sentences from unlabeled set"
                for review in test["description"]:
                    temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_test_reviews.append(" ".join(sum(temp, [])))
            else:

                for review in train["content"]:
                    temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_train_reviews.append(" ".join(sum(temp, [])))
                print "Parsing sentences from unlabeled set"
                for review in test["content"]:
                    temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_test_reviews.append(" ".join(sum(temp, [])))
            # load  model that trained 
            self.model = Word2Vec.load(model_name)
            # self.model = Word2Vec.load(model_name, binary=False)
        return clean_train_reviews, clean_test_reviews
            
        
    # ****** Create average vectors for the training and test sets 加权平均
    def createWord2vec(self, googlenews=False, dataIfSet=False, train=None, test=None, remove_stopwords=False , low_freq_filter=False):

        if dataIfSet:
            self.Word2vecModel(googlenews, dataIfSet, train, test, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter,model_name='./model/BoWWAV')
        else:
            self.Word2vecModel(googlenews, dataIfSet, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter, model_name='./model/BoWWAV')
            # ****** Create average vectors for the training and test sets
        print "Creating average feature vecs for training reviews"
        wvutility = Word2VecUtility()
        self.trainDataVec = wvutility.getAvgFeatureVecs(wvutility.getCleanReviews(self.train, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter), self.model, self.feature)
    
        print "Creating average feature vecs for test reviews"
    
        self.testDataVec = wvutility.getAvgFeatureVecs(wvutility.getCleanReviews(self.test, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter), self.model, self.feature)
    # 将向量前后拼接起来，设置重叠率，运用插值和采样的方式
    def WordsLink(self, overlappingRate, googlenews=False, dataIfSet=False, train=None, test=None, remove_stopwords=False , low_freq_filter=False):
        print "wordLink:1.训练word2vec model"
        if dataIfSet:
            self.Word2vecModel(googlenews, dataIfSet, train, test, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
        else:
            self.Word2vecModel(googlenews, dataIfSet, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
        print "wordLink:2.得到trainVecs,testVecs"
        wvutility = Word2VecUtility()
        trainReviews = wvutility.getCleanReviews(self.train, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
        testReviews = wvutility.getCleanReviews(self.test, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
        wordfeatureNum = self.feature
        wvu = Word2VecUtility()
        self.trainDataVec = wvu.getLinkWordFeatureVecs(trainReviews, self.model, wordfeatureNum, overlappingRate)
        self.testDataVec = wvu.getLinkWordFeatureVecs(testReviews, self.model, wordfeatureNum, overlappingRate)
    # dataIfSet 数据集是否存在 dm=1 dm模型 dm=0 bow模型
    def createDoc2vec(self, dataIfSet=False, train=None, test=None, remove_stopwords=False , low_freq_filter=False, dm=None):
       
        if dataIfSet:
            self.train = train
            self.test = test
        else:
            df = self.data
            train, test = Word2VecUtility.split_train_test(df, test_portion=0.3)
            # list [[ str]]
            self.train = train
            self.test = test
       
        # unsup = test
        """  从本地读取csv 文件建立 train test unsup集
        train  = pd.read_csv('../data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
        test = pd.read_csv('../data/testData.tsv', header=0, delimiter="\t", quoting=3)
        unsup = pd.read_csv('../data/unlabeledTrainData.tsv', header=0,  delimiter="\t", quoting=3 )
        """
        print "Cleaning and labeling all data sets...\n"
        
        train_reviews = Doc2VecUtility_correct.getCleanLabeledReviews(train, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
        test_reviews = Doc2VecUtility_correct.getCleanLabeledReviews(test, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
        # unsup_reviews = Doc2VecUtility_correct.getCleanLabeledReviews(unsup,remove_stopwords=remove_stopwords,low_freq_filter=low_freq_filter)
    
        n_dim = 400
        
        model_dm_name = "%dfeatures_4minwords_10context_dm" % n_dim
        model_dbow_name = "%dfeatures_4minwords_10context_dbow" % n_dim
        
        
        
        if not os.path.exists(model_dm_name) or not os.path.exists(model_dbow_name):
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                                level=logging.INFO)
     
            num_features = n_dim  # Word vector dimensionality
            min_word_count = 1  # Minimum word count, if bigger, some sentences may be missing
            num_workers = 2  # Number of threads to run in parallel
            context = 8  # Context window size
            downsampling = 1e-4  # Downsample setting for frequent words
     
            
            # all_reviews = np.concatenate((train_reviews, test_reviews, unsup_reviews))
            all_reviews = []
            all_reviews.extend(train_reviews)
            all_reviews.extend(test_reviews)
           # reviews_str=[]
           # for reviewLabel in all_reviews:
            #    reviews_str.append(reviewLabel[0])
            
            # all_reviews.append(unsup_reviews)
            print "Training Doc2Vec model..."
            if dm == 1:
                model_dm = Doc2Vec(min_count=min_word_count, window=context, size=num_features, \
                               sample=downsampling, workers=num_workers)
                model_dm.build_vocab(all_reviews)
                for epoch in range(10):
                    # 打乱
                    all_reviews_perm = []
                    listIndex = np.random.permutation(len(train_reviews))
                    train_reviews_perm = []
                    for k in listIndex:                   
                        train_reviews_perm.append(train_reviews[k])
                    train_perm = train.ix[listIndex]
                    
                    listIndex = np.random.permutation(len(test_reviews))
                    test_reviews_perm = []
                    for k in listIndex:                          
                        test_reviews_perm.append(test_reviews[k])
                    test_perm = test.ix[listIndex]    
                    all_reviews_perm.extend(train_reviews_perm)
                    all_reviews_perm.extend(test_reviews_perm)
                    model_dm.train(all_reviews_perm)
                model_dm.init_sims(replace=True)
                model_dm.save(model_dm_name)
                self.model = model_dm
            elif  dm == 0:
                model_dbow = Doc2Vec(min_count=min_word_count, window=context, size=num_features,
                                 sample=downsampling, workers=num_workers, dm=0)
                model_dbow.build_vocab(all_reviews)
                for epoch in range(10):
                    all_reviews_perm = []
                    listIndex = np.random.permutation(len(train_reviews))
                    train_reviews_perm = []
                    for k in listIndex:                   
                        train_reviews_perm.append(train_reviews[k])
                    train_perm = train.ix[listIndex]
                    
                    listIndex = np.random.permutation(len(test_reviews))
                    test_reviews_perm = []
                    for k in listIndex:                          
                        test_reviews_perm.append(test_reviews[k])
                    test_perm = test.ix[listIndex]    
                    all_reviews_perm.extend(train_reviews_perm)
                    all_reviews_perm.extend(test_reviews_perm)
                    model_dbow.train(all_reviews_perm)
                model_dbow.init_sims(replace=True)
                model_dbow.save(model_dbow_name)
                self.model = model_dbow
            else:
                print "dm is not correct"
                raise Exception("dm is not correct,dm is 1 or 0")
 
        elif os.path.exists(model_dm_name):
            Doc2Vec.load(model_dm_name)
        elif os.path.exists(model_dbow_name):
            Doc2Vec.load(model_dbow_name)
        else:
            raise Exception("doc2vec Dm and dbow model are  not exists")
        self.feature = num_features
        self.trainDataVec = np.zeros((len(self.train), num_features))
        self.testDataVec = np.zeros((len(self.test), num_features))
        
        # 将打乱的dataVec和lable 对应起来。并分出测试集与训练集
        self.train = train_perm
        self.test = test_perm
        for i in range(len(train)):
            
            self.trainDataVec[i] = self.model.docvecs[train_reviews_perm[i].tags].reshape(1, self.feature)
            
            
        for i in range(len(test)):
            self.testDataVec[i] = self.model.docvecs[test_reviews_perm[i].tags].reshape(1, self.feature)
            
        
    """  定义 矩阵模型，每一行为篇文本，一列为一个词，把这些文本集的列数规整化"""

    def CreateAutoEncodeModel(self, googlenews=False, dataIfSet=False, train=None, test=None, remove_stopwords=False , low_freq_filter=False, model_name=None):
        
        if dataIfSet:
            self.train = train
            self.test = test
        else:
            df = self.data
            train, test = Word2VecUtility.split_train_test(df, test_portion=0.3)
            # list [[ str]]
            self.train = train
            self.test = test
       
        n_dim = self.feature   
        num_features = n_dim  # Word vector dimensionality
        min_word_count = 5  # Minimum word count
        num_workers = 4  # Number of threads to run in parallel
        context = 10  # Context window size
        downsampling = 1e-3  # Downsample setting for frequent words
        clean_train_reviews = []
        clean_test_reviews = []
        if model_name == None:
            model_name = "%dfeatures_%dminwords_%dcontext" % (n_dim, min_word_count, context)
        if not os.path.exists(model_name):  # 判断是否训练好了模型和已处理
        # 了数据（使用相同的预处理过程对于word2vec模型和clean_train_reviews 用于新的矩阵模型，
            sentences = []  # list of list of sentense  [[],[]] they are treated as input of train word2vec model 
            print "Parsing sentences from training set"
            dumpTrain = []
            colnames = train.columns.tolist()
            if colnames.__contains__("description") == True:
            # pantent 数据集
                for review in train["description"]:
                    temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_train_reviews.append(" ".join(sum(temp, [])))  # temp=[[1,2],[3,4]] sum(temp,[])=[1, 2, 3, 4]
                    dumpTrain.append(" ".join(sum(temp, [])))
                trainOut = pd.DataFrame(data={"IPCNum":train['IPCNum'],"content":dumpTrain})
                trainOut.to_csv("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_patent.csv")
                print "Parsing sentences from unlabeled set"
                dumpTest = []
                for review in test["description"]:
                    temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_test_reviews.append(" ".join(sum(temp, [])))
                    dumpTest.append(" ".join(sum(temp, [])))
                testOut = pd.DataFrame(data={"IPCNum":test['IPCNum'],"content":dumpTest})
                testOut.to_csv("./clean_data/Clean_test_reviews_"+model_name.split("/")[-1]+"_patent.csv")
            else:  #  standford sst dataset
                for review in train["content"]:
                    temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_train_reviews.append(" ".join(sum(temp, [])))  # temp=[[1,2],[3,4]] sum(temp,[])=[1, 2, 3, 4]
                    dumpTrain.append(" ".join(sum(temp, [])))
                trainOut = pd.DataFrame(data={"label":train['label'],"content":dumpTrain})
                trainOut.to_csv("./clean_data/Clean_train_"+model_name.split("/")[-1]+"_reviews.csv")
                print "Parsing sentences from unlabeled set"
                dumpTest = []
                for review in test["content"]:
                    temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_test_reviews.append(" ".join(sum(temp, [])))
                    dumpTest.append(" ".join(sum(temp, [])))
                testOut = pd.DataFrame(data={'label':test['label'],"content":dumpTest})
                testOut.to_csv("./clean_data/Clean_test_"+model_name.split("/")[-1]+"_reviews.csv")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                                level=logging.INFO)

            num_features = n_dim  # Word vector dimensionality
            min_word_count = 1  # Minimum word count
            num_workers = 4  # Number of threads to run in parallel
            context = 10  # Context window size
            downsampling = 1e-3  # Downsample setting for frequent words
     
            print "Training Word2Vec model..."
            print "train words time:", time.gmtime()
            sentencesRepeat = sentences * 20
            self.model = Word2Vec(sentencesRepeat, workers=num_workers, alpha=0.025, sg=1, \
                            size=num_features, min_count=min_word_count, \
                            window=context, sample=downsampling, seed=1)
            self.model.init_sims(replace=True)
            self.model.save(model_name)
            print "end time:", time.gmtime()
        else:
            # load  model that trained 
            # sentences = []
            print "Parsing sentences from training set"
            if dataIfSet:
                if os.path.exists("./clean_data/Clean_train_"+model_name.split("/")[-1]+"_reviews.csv"):
                    print "standford sst dataset of cleaned data be readed :"
                    train_df = pd.read_csv("./clean_data/Clean_train_"+model_name.split("/")[-1]+"_reviews.csv")
                    self.train['content']=train_df['content']
                    self.train['label']=train_df['label']
                    train=self.train
                    print "Parsing sentences from unlabeled set"
                    test_df = pd.read_csv("./clean_data/Clean_test_"+model_name.split("/")[-1]+"_reviews.csv")
                    self.test['content']=test_df['content']
                    self.test['label']=test_df['label']
                    test=self.test
                else:
                    dumpTrain = []
                    for review in train["content"]:
                        temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                        clean_train_reviews.append(" ".join(sum(temp, [])))  # temp=[[1,2],[3,4]] sum(temp,[])=[1, 2, 3, 4]
                        dumpTrain.append(" ".join(sum(temp, [])))
                    train_df = pd.DataFrame(data={"label":train['label'],"content":dumpTrain})
                    train_df.to_csv("./clean_data/Clean_train_"+model_name.split("/")[-1]+"_reviews.csv")
                    print "Parsing sentences from unlabeled set"
                    dumpTest = []
                    for review in test["content"]:
                        temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                        clean_test_reviews.append(" ".join(sum(temp, [])))
                        dumpTest.append(" ".join(sum(temp, [])))
                    test_df = pd.DataFrame(data={'label':test['label'],"content":dumpTest})
                    test_df.to_csv("./clean_data/Clean_test_"+model_name.split("/")[-1]+"_reviews.csv")
            else:
                if os.path.exists("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_patent.csv"):
                    print "patent dataset of cleaned data be readed :"
                    train_df = pd.read_csv("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_patent.csv")
                    self.train['description']=train_df['content']
                    self.train['IPCNum']=train_df['IPCNum']
                    train=self.train
                    print "Parsing sentences from unlabeled set"
                    test_df = pd.read_csv("./clean_data/Clean_test_reviews_"+model_name.split("/")[-1]+"_patent.csv")
                    self.test['description']=test_df['content']
                    self.test['IPCNum']=test_df['IPCNum']
                    test=self.test
                else:
                    dumpTrain = []
                    for review in train["description"]:
                        temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                        clean_train_reviews.append(" ".join(sum(temp, [])))  # temp=[[1,2],[3,4]] sum(temp,[])=[1, 2, 3, 4]
                        dumpTrain.append(" ".join(sum(temp, [])))
                    train_df = pd.DataFrame(data={"IPCNum":train["IPCNum"],"content":dumpTrain})
                    train_df.to_csv("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_patent.csv")
                    print "Parsing sentences from unlabeled set"
                    dumpTest = []
                    for review in test["description"]:
                        temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                        clean_test_reviews.append(" ".join(sum(temp, [])))
                        dumpTest.append(" ".join(sum(temp, [])))
                    test_df = pd.DataFrame(data={"IPCNum":test["IPCNum"],"content":dumpTest})
                    test_df.to_csv("./clean_data/Clean_test_reviews_"+model_name.split("/")[-1]+"_patent.csv")
            for cont in train_df["content"]:
                if not isinstance(cont, str) and math.isnan(cont):
                    clean_train_reviews.append("")
                else:
                    clean_train_reviews.append(cont)
            """
            for review in train["description"]:
                temp=Word2VecUtility().contents_to_sentences(review)
                sentences+=temp
                clean_train_reviews.append(" ".join(sum(temp,[])))
            """
            for cont in test_df["content"]:
                if not isinstance(cont, str) and math.isnan(cont):
                    clean_test_reviews.append("")
                else:
                    clean_test_reviews.append(cont)
            """
            for review in test["description"]:
                temp=Word2VecUtility().contents_to_sentences(review)
                sentences += temp
                clean_test_reviews.append(" ".join(sum(temp,[])))
            """
            self.model = Word2Vec.load(model_name)
            # print self.model
        print "Creating AutoEncoderModel vecs for reviews"
        wvu = Word2VecUtility()
        model = self.model  #  300features_40minwords_10context
        
       # clean_train_reviews = Word2VecUtility.getCleanReviewsStr(train)
     #   clean_test_reviews = Word2VecUtility.getCleanReviewsStr(test)

        line_avg_words = 0
        short_words = 5000
        short_review = None
        for review in clean_train_reviews:
            if short_words > len(review.split(" ")):
                short_words = len(review.split(" "))
                short_review = review
            line_avg_words += len(review.split(" "))
        line_avg_words /= len(clean_train_reviews)
        print short_review
        # train_feature :list each item is like[Feature(100, 40, 22),Feature(32, 190, 150), Feature(2, 100, 100)]
        train_Features_matrix = wvu.getAllWordsMatrix(clean_train_reviews, model, num_features, line_avg_words, short_words, policy='DA')
        test_Features_matrix = wvu.getAllWordsMatrix(clean_test_reviews, model, num_features, line_avg_words, short_words, policy="DA")
        self.trainDataVec = train_Features_matrix
        self.testDataVec = test_Features_matrix
        return train_Features_matrix, test_Features_matrix
   
    def createBagofWordConW2V(self, googlenews=False, tfidfset=True, dataIfSet=False, train=None, test=None, selectF=True, selectMax=True, remove_stopwords=False , low_freq_filter=False, fs_method='IG', fs_num=5000, model_name=""):
        # df=textParserRead().getFramdata()
     
        if dataIfSet:
            self.train = train
            self.test = test
        else:
            df = self.data
            train, test = Word2VecUtility.split_train_test(df, test_portion=0.3)
            # list [[ str]]
            self.train = train
            self.test = test
        clean_train_reviews = []
        clean_test_reviews = []
        # doc_terms_list_train=[]
        # 训练词向量
        n_dim = self.feature   
        num_features = n_dim  # Word vector dimensionality
        min_word_count = 5  # Minimum word count
        num_workers = 4  # Number of threads to run in parallel
        context = 10  # Context window size
        downsampling = 1e-3  # Downsample setting for frequent words
        if  model_name == "":
            
            # model_name="G:\worksp\pythonScrapy\src\GoogleNews-vectors-negative300.bin"
            model_name = "%dfeatures_%dminwords_%dcontext" % (n_dim, min_word_count, context)
        if not os.path.exists(model_name):  # 不存在模型，训练模型并把预处理的数据保存
            sentences = []
            print "Parsing sentences from training set"
            
            dumpTrain = []
            dumpTest = []
            colnames = train.columns.tolist()
            if colnames.__contains__("description") == True:
            # if train.get("description",None)!=None:#pantent 数据集
                for review in train["description"]:
                    temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_train_reviews.append(" ".join(sum(temp, [])))  # temp=[[1,2],[3,4]] sum(temp,[])=[1, 2, 3, 4]
                    dumpTrain.append(" ".join(sum(temp, [])))
                    # doc_terms_list_train.append(sum(temp,[]))
                trainOut = pd.DataFrame(data={"IPCNum":train["IPCNum"],"content":dumpTrain})
                trainOut.to_csv("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_patent.csv")
               
                print "Parsing sentences from unlabeled set"
                 
                for review in test["description"]:
                    temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_test_reviews.append(" ".join(sum(temp, [])))
                    dumpTest.append(" ".join(sum(temp, [])))
                    
                testOut = pd.DataFrame(data={"IPCNum":test["IPCNum"],"content":dumpTest})
                testOut.to_csv("./clean_data/Clean_test_reviews_"+model_name.split("/")[-1]+"_patent.csv")    
            else:
                for review in train["content"]:  # standford sst数据集
                    temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_train_reviews.append(" ".join(sum(temp, [])))  # temp=[[1,2],[3,4]] sum(temp,[])=[1, 2, 3, 4]
                    dumpTrain.append(" ".join(sum(temp, [])))
                    # doc_terms_list_train.append(sum(temp,[]))
                trainOut = pd.DataFrame(data={"label":train['label'], "content":dumpTrain})
                trainOut.to_csv("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_sst.csv") 
                print "Parsing sentences from unlabeled set"
     
                for review in test["content"]:
                    temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    sentences += temp
                    clean_test_reviews.append(" ".join(sum(temp, [])))
                    dumpTest.append(" ".join(sum(temp, [])))
                testOut = pd.DataFrame(data={"label":test['label'], "content":dumpTest})
                testOut.to_csv("./clean_data/Clean_test_reviews_"+model_name.split("/")[-1]+"_sst.csv")   
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                                level=logging.INFO)
            num_features = n_dim  # Word vector dimensionality
            min_word_count = 5  # Minimum word count
            num_workers = 4  # Number of threads to run in parallel
            context = 10  # Context window size
            downsampling = 1e-3  # Downsample setting for frequent words
            print "Training Word2Vec model..."
            # 训练数据重复 20次
            sentencesRepeat = sentences * 20
            self.model = Word2Vec(sentencesRepeat, workers=num_workers, alpha=0.025, sg=1, \
                            size=num_features, min_count=min_word_count, \
                            window=context, sample=downsampling, seed=1)
            self.model.init_sims(replace=True)
            self.model.save(model_name)
        else:
            # load  model that trained 
            dumpTrain = []
            dumpTest = []
            # 对于sst数据集
            if os.path.exists("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_sst.csv") and dataIfSet:
                print "standford sst dataset of cleaned data be readed :"
                train_df = pd.read_csv("./clean_data/Clean_train_reviews_BowC_sst.csv")
                self.train['label']=train_df['label']
                
                self.train['content']=train_df['content']
                train=self.train
                print "Parsing sentences from unlabeled set"
                test_df = pd.read_csv("./clean_data/Clean_test_reviews_"+model_name.split("/")[-1]+"_sst.csv")
                self.test['label']=test_df['label']
                
                self.test['content']=test_df['content']
                test=self.test
            elif dataIfSet and not os.path.exists("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_sst.csv"):
                for review in train["content"]:  # standford sst数据集
                    temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    clean_train_reviews.append(" ".join(sum(temp, [])))  # temp=[[1,2],[3,4]] sum(temp,[])=[1, 2, 3, 4]
                    dumpTrain.append(" ".join(sum(temp, [])))
                    # doc_terms_list_train.append(sum(temp,[]))
                train_df = pd.DataFrame(data={"label":train['label'], "content":dumpTrain})
                train_df.to_csv("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_sst.csv") 
                print "Parsing sentences from unlabeled set"
                print "Parsing sentences from unlabeled set"
                for review in test["content"]:
                    temp = Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                
                    clean_test_reviews.append(" ".join(sum(temp, [])))
                    dumpTest.append(" ".join(sum(temp, [])))
                test_df = pd.DataFrame(data={"label":test['label'], "content":dumpTest})
                test_df.to_csv("./clean_data/Clean_test_reviews_"+model_name.split("/")[-1]+"_sst.csv") 
            # 对于patent数据集
            elif os.path.exists("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_patent.csv") and not dataIfSet:
                print "patent dataset of cleaned data be readed :"
                train_df = pd.read_csv("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_patent.csv")
                self.train['IPCNum']=train_df['IPCNum']
                
                self.train['description']=train_df['content']
                train=self.train
                print "Parsing sentences from unlabeled set"
                test_df = pd.read_csv("./clean_data/Clean_test_reviews_"+model_name.split("/")[-1]+"_patent.csv")
                self.test['IPCNum']=test_df['IPCNum']
                self.test['description']=test_df['content']
                test=self.test
            else:
                for review in train["description"]:
                    temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
            
                    clean_train_reviews.append(" ".join(sum(temp, [])))  # temp=[[1,2],[3,4]] sum(temp,[])=[1, 2, 3, 4]
                    dumpTrain.append(" ".join(sum(temp, [])))
                    # doc_terms_list_train.append(sum(temp,[]))
                train_df = pd.DataFrame(data={"IPCNum":train["IPCNum"],"content":dumpTrain})
                train_df.to_csv("./clean_data/Clean_train_reviews_"+model_name.split("/")[-1]+"_patent.csv")
               
                print "Parsing sentences from unlabeled set"
                 
                for review in test["description"]:
                    temp = Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                 
                    clean_test_reviews.append(" ".join(sum(temp, [])))
                    dumpTest.append(" ".join(sum(temp, [])))
                    
                test_df = pd.DataFrame(data={"IPCNum":test["IPCNum"],"content":dumpTest})
                test_df.to_csv("./clean_data/Clean_test_reviews_"+model_name.split("/")[-1]+"_patent.csv") 
            for cont in train_df["content"]:
                if not isinstance(cont, str) and math.isnan(cont):
                    clean_train_reviews.append("")
                else:
                    clean_train_reviews.append(cont)
            """
            for review in train["description"]:
                temp=Word2VecUtility().contents_to_sentences(review)
                sentences+=temp
                clean_train_reviews.append(" ".join(sum(temp,[])))
            """
            for cont in test_df["content"]:
                if not isinstance(cont, str) and math.isnan(cont):
                    clean_test_reviews.append("")
                else:
                    clean_test_reviews.append(cont)
            self.model = Word2Vec.load(model_name)   
            # self.model=Word2Vec.load_word2vec_format("G:\worksp\pythonScrapy\src\GoogleNews-vectors-negative300.bin",binary=True)
        
        # clean_train_reviews = Word2VecUtility().getCleanReviewsStr(train) 用reviews_to_sentense处理，和训练word2vec模型时预处理结合了
        # clean_test_reviews = Word2VecUtility().getCleanReviewsStr(test)
        vectorizer = None
        if not tfidfset:        
            vectorizer = CountVectorizer(analyzer="word", \
                             tokenizer=None, \
                             preprocessor=None, \
                             stop_words=None, \
                             max_features=5000)    
        else:
            vectorizer = TfidfVectorizer()
        # select Feature :
        if selectF == True:
            
            doc_str_list_train = clean_train_reviews
            doc_str_list_test = clean_test_reviews
            colna = train.columns.tolist()
            if colna.__contains__("IPCNum") == True:
                doc_class_list_train = train['IPCNum'].tolist()
                temp = []
                for classLabel in doc_class_list_train:
                    temp.append(classLabel[:3])
                doc_class_list_train = temp
                
                
                doc_class_list_test = test['IPCNum'].tolist()
                temp = []
                for classLabel in doc_class_list_test:
                    temp.append(classLabel[:3])
                doc_class_list_test = temp
            else:
                doc_class_list_train = train['label'].tolist()
                doc_class_list_test = test['label'].tolist()
            doc_terms_list_train = [word.split() for word in clean_train_reviews]
            doc_terms_list_test = [word.split() for word in clean_test_reviews]
            train_data_features, test_data_features, vectorizer = feature_select_help(fs_method, \
                        fs_num, doc_str_list_train, \
                        doc_str_list_test, doc_class_list_train, \
                doc_class_list_test, doc_terms_list_train, doc_terms_list_test)
       
            # Feature Select end;
        else:
            # train_data_features = vectorizer.fit_transform(clean_train_reviews)
        # Numpy arrays are easy to work with, so convert the result to an
        # array
           # train_data_features = train_data_features.toarray()
           # test_data_features = vectorizer.transform(clean_test_reviews)
           # test_data_features = test_data_features.toarray()
            
            data_features = vectorizer.fit_transform(clean_train_reviews + clean_test_reviews).toarray()
            train_data_features = data_features[:len(clean_train_reviews)]
            test_data_features = data_features[len(clean_train_reviews):]
        self.trainDataVec = train_data_features
        self.testDataVec = test_data_features
        
        
        # 为值为0的添加其他值
        vocAndIndex = vectorizer.vocabulary_
        vocAndIndex_unicodeTostr={}
        for item in vocAndIndex:
            vocAndIndex_unicodeTostr[item.encode("utf-8")]=vocAndIndex[item]
        vocAndIndex=vocAndIndex_unicodeTostr
        # 训练集矩阵修改,对于每个词汇表中词如果，它的值接近0 那么把它表示为 相似的词的值*相似性
        print "trainvec scan："
        cacheSimlarList = {}
        for row in range(self.trainDataVec.shape[0]):  # 每一行为一个样本
            rowStart = time.clock()
            newdic = {}
            newdicCount = {}
            for w in clean_train_reviews[row].split(" "):  # 对于每个样本中的每个词
                if w in vocAndIndex:  # 假如当前词在词汇表中，并且词向量存在
                    if cacheSimlarList.has_key(w):
                        sim = cacheSimlarList[w]
                    else:
                        sim = self.findSimilar(w, self.model)  # if w in self.model.vocab,查找相近词和相似性
                        if sim == None:  # model.vocab中不存在w
                            continue
                        cacheSimlarList[w] = sim  # 将 w：sim添加到dic（cacheSimlarList中）
                    
                    for simword, simval in sim:  # 对于每个相似 词做处理
                        if simval < 0.8:
                            break
                        if simval > 0.8 and simword not in clean_train_reviews[row].split(" ") and simword in vocAndIndex and \
                        self.trainDataVec[row, vocAndIndex[simword]] == 0.:  # 相似性达到阈值，并且相似词不在train中，在词表中，并且权重值为0
                            print "row", row, " col:", vocAndIndex[w], " changed by "
                            print "row", row, " col:", vocAndIndex[simword]
                            if selectMax == True:
                                if newdic.has_key(simword):
                                    if newdic[simword] < simval * self.trainDataVec[row, vocAndIndex[w]]:
                                        newdic[simword] = simval * self.trainDataVec[row, vocAndIndex[w]]
                                else:
                                    newdic[simword] = simval * self.trainDataVec[row, vocAndIndex[w]]
                            else:
                                if newdic.has_key(simword):  # 如果有其他词的相似词列表中含有该词 继续加权相加
                                    newdic[simword] += simval * self.trainDataVec[row, vocAndIndex[w]]
                           
                                    newdicCount[simword] += 1
                                else:
                                    newdic[simword] = simval * self.trainDataVec[row, vocAndIndex[w]]
                                    newdicCount[simword] = 1
            # if selectMax!=True:
            if len(newdic) > 0:  # 对于相似词，更新trainDataVec矩阵
                for word in newdic:  
                    if selectMax==True:
                        self.trainDataVec[row, vocAndIndex[word]] = newdic[word]
                    else:
                        self.trainDataVec[row, vocAndIndex[word]] = newdic[word] / newdicCount[word]
            rowEnd = time.clock()
            # print "update row time cost:",rowEnd-rowStart
      

        # 对于测试集矩阵的处理                      
        print "testvec scan："    
        for row in range(self.testDataVec.shape[0]):  # 每一行为一个样本
            rowStart = time.clock()
            newdic = {}
            newdicCount = {}
            for w in clean_test_reviews[row].split(" "):  # 对于每个样本中的每个词
                if w in vocAndIndex:  # 假如当前词在词汇表中，并且词向量存在
                    if cacheSimlarList.has_key(w):
                        sim = cacheSimlarList[w]
                    else:
                        sim = self.findSimilar(w, self.model)  # if w in self.model.vocab,查找相近词和相似性
                        if sim == None:  # model.vocab中不存在w
                            continue
                        cacheSimlarList[w] = sim  # 将 w：sim添加到dic（cacheSimlarList中）
                    
                    for simword, simval in sim:  # 对于每个相似 词做处理
                        if simval < 0.8:
                            break
                        if simval > 0.8 and simword not in clean_test_reviews[row].split(" ") \
                        and simword in vocAndIndex and self.testDataVec[row, vocAndIndex[simword]] == 0.:  # 相似性达到阈值，并且相似词不在train中，在词表中，并且权重值为0
                            print "row", row, " col:", vocAndIndex[w], " changed by "
                            print "row", row, " col:", vocAndIndex[simword]
                            if selectMax == True:
                                if newdic.has_key(simword): 
                                    if newdic[simword] < simval * self.testDataVec[row, vocAndIndex[w]]:
                                        newdic[simword] = simval * self.testDataVec[row, vocAndIndex[w]]
                                else:	
                                    newdic[simword] = simval * self.testDataVec[row, vocAndIndex[w]]
                                    newdicCount[simword] = 1
                               
                            else:
                                if newdic.has_key(simword):  # 如果有其他词的相似词列表中含有该词 继续加权相加
                                    newdic[simword] += simval * self.testDataVec[row, vocAndIndex[w]]
                                    newdicCount[simword] += 1
                                else:
                                    newdic[simword] = simval * self.testDataVec[row, vocAndIndex[w]]
                                    newdicCount[simword] = 1
            # if selectMax!=True:
            if len(newdic) > 0:  # 对于相似词，更新trainDataVec矩阵
                for word in newdic:
                    if selectMax:
                        self.testDataVec[row, vocAndIndex[word]] = newdic[word]
                    else:  
                        self.testDataVec[row, vocAndIndex[word]] = newdic[word] / newdicCount[word]
            rowEnd = time.clock()
           # print "update row time cost:",rowEnd-rowStart            
        # old version time cost to much,新版本把循环调节位置，判断条件，修改添加cacheSimlarlist 保存{w：simList}相似值词典
#        for row in range(self.trainDataVec.shape[0]):#每个样本
#            rstart=time.clock()
#            for key in vocAndIndex:#词汇表中的每个词（每一列）
#                if self.trainDataVec[row,vocAndIndex[key]]==0.:#当前列的权重值为0
#                    weight=0
#                    count=0
#                    start=time.clock()
#                    sim=self.findSimilar(key,self.model)# 找到当前值的相近的词，它们中在词汇表中和样本中
#                    end=time.clock()
#                    print "findSimllar function time cost:",end-start
#                    if sim==None:
#                        continue
#                    
#                    for simword,simval in sim:#在样本词汇表中 更新 值（使用计算的相似词的权重更新）
#                        if simword in clean_train_reviews[row] and simword in vocAndIndex:
#                            print "row",row," col:",vocAndIndex[key]," changed by "
#                            print "row",row," col:",vocAndIndex[simword]
#                            if sim>0.8 and count<5:
#                                count=count+1
#                                weight+=simval*self.trainDataVec[row,vocAndIndex[simword]]
#                            else:
#                                break
#                    end2=time.clock()
#                    print "update all time cost:",end2-end
#                    if count==0:
#                        weight=0.#考虑是否用周围的词表示
#                    else:
#                        weight=weight/count
#                    self.trainDataVec[row,vocAndIndex[key]]=weight
#            rend=time.clock()
#            print "update row time cost:",rend-rstart
#        # 测试集矩阵修改
#        print "testvec scan："
#        for row in range(self.testDataVec.shape[0]):
#            newdic={}
#            newdicCount={}
#            for w in clean_test_reviews[row]:
#                sim=self.findSimilar(w,self.model)
#                if sim==None:
#                    continue
#                if w not in vocAndIndex:                    
#                    continue# 考虑增加它相似词的权重
#                else:#找它的相似词并且权重为0的，修改它们的值
#                    for simword,simval in sim:
#                        if simval>0.7 and simword  in vocAndIndex and self.testDataVec[row,vocAndIndex[simword]]<0.0001:
#                            print "changed"
#                            print "row",row," col:",vocAndIndex[simword]
#                            if newdic.has_key(simword):
#                                newdic[simword]+= simval*self.testDataVec[row,vocAndIndex[w]]
#                                newdicCount[simword]+=1
#                            else:
#                                newdic[simword]=simval*self.testDataVec[row,vocAndIndex[w]]
#                                newdicCount[simword]=1
#            for word in newdic:  
#                self.testDataVec[row,vocAndIndex[word]]=newdic[word]/newdicCount[word]
        
    def findSimilar(self, key, model):
        if model.vocab.has_key(key):
            simlarlis = model.most_similar(key)
        else:
            simlarlis = None
        return simlarlis

        
    """   bag of word model"""    
    def createBagofWord(self, tfidfset=True, dataIfSet=False, train=None, test=None, selectF=True, fs_method='IG', fs_num=5000, remove_stopwords=False , low_freq_filter=False):
        # df=textParserRead().getFramdata()
        
        if dataIfSet:
            self.train = train
            self.test = test
        else:
            df = self.data
            train, test = Word2VecUtility.split_train_test(df, test_portion=0.3)
            # list [[ str]]
            self.train = train
            self.test = test
        clean_train_reviews = Word2VecUtility().getCleanReviewsStr(train, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
        clean_test_reviews = Word2VecUtility().getCleanReviewsStr(test, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
        doc_terms_list_train = [train_each_str.split(" ")  for train_each_str in clean_train_reviews]
        doc_terms_list_test = [test_each_str.split(" ")  for test_each_str in clean_test_reviews]
        if not tfidfset:        
            vectorizer = CountVectorizer(analyzer="word", \
                             tokenizer=None, \
                             preprocessor=None, \
                             stop_words=None, \
                             max_features=5000)    
        else:
            vectorizer = TfidfVectorizer()
            
        # select Feature :
        if selectF == True:
         
            doc_str_list_train = clean_train_reviews
            doc_str_list_test = clean_test_reviews
            colna = train.columns.tolist()
            if colna.__contains__("IPCNum") == True:
                
                doc_class_list_train = train['IPCNum'].tolist()
                temp = []
                for classLabel in doc_class_list_train:
                    temp.append(classLabel[:3])
                doc_class_list_train = temp
                
                doc_class_list_test = test['IPCNum'].tolist()
                temp = []
                for classLabel in doc_class_list_test:
                    temp.append(classLabel[:3])
                doc_class_list_test = temp
            else:
                doc_class_list_train = train['label'].tolist()
                doc_class_list_test = test['label'].tolist()
        
            train_data_features, test_data_features, vectorizer = feature_select_help(fs_method, fs_num, \
                    doc_str_list_train, doc_str_list_test, \
                    doc_class_list_train, doc_class_list_test, doc_terms_list_train, doc_terms_list_test)
       
            # Feature Select end;
        else:
            data_features = vectorizer.fit_transform(clean_train_reviews + clean_test_reviews).toarray()
            train_data_features = data_features[:len(clean_train_reviews)]
            test_data_features = data_features[len(clean_train_reviews):]
        # Numpy arrays are easy to work with, so convert the result to an
        # array
        # train_data_features = train_data_features.toarray()
        #  test_data_features = vectorizer.transform(clean_test_reviews)
            # test_data_features = test_data_features.toarray()
        
        for i in range(len(train)):
            train['']
        self.trainDataVec = train_data_features
        self.testDataVec = test_data_features
        
    # 使用自己构造的特征选择 bag of word model
    def createBOWUseNewFeatureExtract(self, tfidfset=True, dataIfSet=False, train=None, test=None, remove_stopwords=False , selectF=False, low_freq_filter=False, fs_method='IG', fs_num=5000, similarVal=0.6):
        # df=textParserRead().getFramdata()
        
        clean_train_reviews, clean_test_reviews = self.Word2vecModel(dataIfSet=dataIfSet, train=train,
                                test=test, remove_stopwords=remove_stopwords , low_freq_filter=low_freq_filter, model_name="./model/BOWFeatureExtract_model")
        
        # 用IG', 'MI', 'WLLR'特征选择
        doc_terms_list_train = [train_each_str.split(" ")  for train_each_str in clean_train_reviews]
        doc_terms_list_test = [test_each_str.split(" ")  for test_each_str in clean_test_reviews]
        doc_str_list_train = clean_train_reviews
        doc_str_list_test = clean_test_reviews
        colna = self.train.columns.tolist()
        if colna.__contains__("IPCNum") == True:
            doc_class_list_train = self.train['IPCNum'].tolist()
            
            temp = []
            for classLabel in doc_class_list_train:
                temp.append(classLabel[:3])
            doc_class_list_train = temp
                
                
                
            doc_class_list_test = self.test['IPCNum'].tolist()
            temp = []
            for classLabel in doc_class_list_test:
                temp.append(classLabel[:3])
            doc_class_list_test = temp
        else:
            doc_class_list_train = self.train['label'].tolist()
            doc_class_list_test = self.test['label'].tolist()
        if selectF == True:
            feature_terms = feature_select_termsets(fs_method, fs_num, \
                                                  doc_str_list_train, doc_str_list_test, \
            doc_class_list_train, doc_class_list_test, doc_terms_list_train, doc_terms_list_test)
        else:
            # 当feature-terms=None FeatureExtractsimiliarCon 不做特征提取
            feature_terms = None
        # 词向量模型
        modelWord = self.model
        # 在新的特征选择后的模型词库表中，从新运用特征合并
        corpusData = clean_train_reviews + clean_test_reviews
        data_features = FeatureExtractSimilarCon(modelWord, corpus=corpusData, term_set_fs=feature_terms, similarVal=similarVal)        
        # 划分测试和训练向量矩阵
        train_data_features = data_features[:len(clean_train_reviews)]
        test_data_features = data_features[len(clean_train_reviews):]
       
        # array
        # train_data_features = train_data_features.toarray()
        #  test_data_features = vectorizer.transform(clean_test_reviews)
            # test_data_features = test_data_features.toarray()
        self.trainDataVec = train_data_features
        self.testDataVec = test_data_features
        
        
    """
        # 调用方式
        def distance(f1, f2):
            return sqrt( (f1.x - f2.x)**2  + (f1.y - f2.y)**2 + (f1.z - f2.z)**2 )
        def main():
            features1 = [Feature(100, 40, 22), Feature(211, 20, 2),
                         Feature(32, 190, 150), Feature(2, 100, 100)]
            weights1  = [0.4, 0.3, 0.2, 0.1]
            features2 = [Feature(0, 0, 0), Feature(50, 100, 80), Feature(255, 255, 255)]
            weights2  = [0.5, 0.3, 0.2]    
            print emd( (features1, weights1), (features2, weights2), distance )
    """
    def createWMD(self, googlenews=False, dataIfSet=False, train=None, test=None, remove_stopwords=False, low_freq_filter=False, model_name=""):
        """  从本地读取csv 文件建立 train test unsup集
        train  = pd.read_csv('../data/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
        #test = pd.read_csv('../data/testData.tsv', header=0, delimiter="\t", quoting=3)
        unsup = pd.read_csv('../data/unlabeledTrainData.tsv', header=0,  delimiter="\t", quoting=3 )
        """
        if dataIfSet:
            self.train = train
            self.test = test
        else:
            df = self.data
            train, test = Word2VecUtility.split_train_test(df, test_portion=0.3)
            # list [[ str]]
            self.train = train
            self.test = test
        n_dim = self.feature   
        num_features = n_dim  # Word vector dimensionality
        min_word_count = 5  # Minimum word count
        num_workers = 4  # Number of threads to run in parallel
        context = 10  # Context window size
        downsampling = 1e-3  # Downsample setting for frequent words
        if  model_name == "":
            model_name = "%dfeatures_%dminwords_%dcontext" % (n_dim, min_word_count, context)
            if not os.path.exists(model_name): 
                sentences = []
                print "Parsing sentences from training set"
                colnames = train.columns.tolist()
                if colnames.__contains__("description") == True:
                # if train.get("description",None)!=None:
                    for review in train["description"]:
                        sentences += Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    print "Parsing sentences from unlabeled set"
                    for review in test["description"]:
                        sentences += Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                else:
                    for review in train["content"]:
                        sentences += Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                    print "Parsing sentences from unlabeled set"
                    for review in test["content"]:
                        sentences += Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                
                logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                                    level=logging.INFO)

                num_features = n_dim  # Word vector dimensionality
                min_word_count = 5  # Minimum word count
                num_workers = 4  # Number of threads to run in parallel
                context = 10  # Context window size
                downsampling = 1e-3  # Downsample setting for frequent words
         
                print "Training Word2Vec model..."
                sentenceRepeat = sentences * 20
                self.model = Word2Vec(sentenceRepeat, workers=num_workers, alpha=0.025, sg=1, \
                                size=num_features, min_count=min_word_count, \
                                window=context, sample=downsampling, seed=1)
                self.model.init_sims(replace=True)
                self.model.save(model_name)
            else:
                # load  model that trained 
                self.model = Word2Vec.load(model_name)
                # print self.model
        print "Creating WMD vecs for reviews"
        wvu = Word2VecUtility()
        model = self.model  #  300features_40minwords_10context
        """ 以tifidf 为特征权重，计算特征矩阵 weight"""
        clean_train_reviews = Word2VecUtility().getCleanReviewsStr(train, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
        clean_test_reviews = Word2VecUtility().getCleanReviewsStr(test, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
        
        train_weights = wvu.getFeatureWeight(clean_train_reviews, model)
        test_weights = wvu.getFeatureWeight(clean_test_reviews, model)
        """ fit need trainvecs , distance is already precomputed ,and the the param fit ,can be any forms """
        vectorizer = TfidfVectorizer(min_df=1)
        train_data_feat = vectorizer.fit_transform(clean_train_reviews)
        # Numpy arrays are easy to work with, so convert the result to an
        # array
        train_data_Vec = train_data_feat.toarray()
        self.testDataVec = train_data_Vec
        
        
        """ 计算distance matric  ，
        #每一行为一段文字的feature向量（features1 = [Feature(100, 40, 22),
        Feature(32, 190, 150), Feature(2, 100, 100)]）"""
        # train_feature :list each item is like[Feature(100, 40, 22),Feature(32, 190, 150), Feature(2, 100, 100)]
        train_Features = wvu.getAllFeatureVecs(clean_train_reviews, model, num_features)
        test_Features = wvu.getAllFeatureVecs(clean_test_reviews, model, num_features)
        distanceMatric = wvu.getDistanceMatric(train_Features, test_Features, train_weights, test_weights)
        return distanceMatric
   
    
        
    """词质心模型：WordBogofCenter 2.Create"""
    def create_bag_of_centroids(self, wordlist, word_centroid_map):
        #
        # The number of clusters is equal to the highest cluster index
        # in the word / centroid map
        num_centroids = max(word_centroid_map.values()) + 1
        #
        # Pre-allocate the bag of centroids vector (for speed)
        bag_of_centroids = np.zeros(num_centroids, dtype="float32")
        #
        # Loop over the words in the review. If the word is in the vocabulary,
        # find which cluster it belongs to, and increment that cluster count
        # by one
        for word in wordlist:
            if word in word_centroid_map:
                index = word_centroid_map[word]
                bag_of_centroids[index] += 1
        #
        # Return the "bag of centroids"
        return bag_of_centroids


    """ 对词库聚类，找到每个词心，让这个维度的次数加1  词质心模型：WordBogofCenter 1.getWordCenter"""
    def getWord2vecBagofCenter(self, dataIfSet=False, train=None, test=None, remove_stopwords=False, low_freq_filter=False, model_name=None):
        # df=textParserRead().getFramdata()
    
        if dataIfSet:
            self.train = train
            self.test = test
        else:
            df = self.data
            train, test = Word2VecUtility.split_train_test(df, test_portion=0.3)
            # list [[ str]]
            self.train = train
            self.test = test
       
        n_dim = self.feature
        num_features = n_dim  # Word vector dimensionality
        min_word_count = 5  # Minimum word count
        num_workers = 4  # Number of threads to run in parallel
        context = 10  # Context window size
        downsampling = 1e-3  # Downsample setting for frequent words
        if model_name == None:
            model_name = "%dfeatures_%dminwords_%dcontext" % (n_dim, min_word_count, context)
        
      
        if not os.path.exists(model_name): 
            sentences = []
            print "Parsing sentences from training set"
            colnames = train.columns.tolist()
            if colnames.__contains__("description") == True:
            # if train.get("description",None)!=None:
                for review in train["description"]:
                    sentences += Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                print "Parsing sentences from unlabeled set"
                for review in test["description"]:
                    sentences += Word2VecUtility().contents_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
            else:
                for review in train["content"]:
                    sentences += Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
                print "Parsing sentences from unlabeled set"
                for review in test["content"]:
                    sentences += Word2VecUtility().review_to_sentences(review, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                                level=logging.INFO)

        
            print "Training Word2Vec model..."
            sentencesRepeate = sentences * 20
            self.model = Word2Vec(sentencesRepeate, workers=num_workers, alpha=0.025, sg=1, \
                                size=num_features, min_count=min_word_count, \
                                window=context, sample=downsampling, seed=1)
         
            self.model.init_sims(replace=True)
            self.model.save(model_name)
        else:
            # load  model that trained 
            self.model = Word2Vec.load(model_name)
            # self.model=Word2Vec.load_word2vec_format('/10features_5minwords_5context',binary=True)
        # ****** Create average vectors for the training and test sets
        print "Creating average feature vecs for training reviews"
        model = self.model  #  300features_40minwords_10context
        # ****** Run k-means on the word vectors and print a few clusters
        start = time.time()  # Start time
        # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
        # average of 5 words per cluster
        word_vectors = model.syn0
        num_clusters = word_vectors.shape[0] / 5
    
        # Initalize a k-means object and use it to extract centroids
        print "Running K means"
        kmeans_clustering = KMeans(n_clusters=num_clusters)
        idx = kmeans_clustering.fit_predict(word_vectors)
        # Get the end time and print how long the process took
        end = time.time()
        elapsed = end - start
        print "Time taken for K Means clustering: ", elapsed, "seconds."
    
        # Create a Word / Index dictionary, mapping each vocabulary word to
        # a cluster number
        word_centroid_map = dict(zip(model.index2word, idx))
        # Print the first ten clusters
        for cluster in xrange(0, 10):
            #
            # Print the cluster number
            print "\nCluster %d" % cluster
            #
            # Find all of the words for that cluster number, and print them out
            words = []
            for i in xrange(0, len(word_centroid_map.values())):
                if(word_centroid_map.values()[i] == cluster):
                    words.append(word_centroid_map.keys()[i])
            # print words

        # Create clean_train_reviews and clean_test_reviews as we did before
        #
    
        print "Cleaning training reviews"
    
        clean_train_reviews = Word2VecUtility().getCleanReviews(train, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
        
        print "Cleaning test reviews"
    
        clean_test_reviews = Word2VecUtility().getCleanReviews(test, remove_stopwords=remove_stopwords, low_freq_filter=low_freq_filter)
  
        # ****** Create bags of centroids
        #
        # Pre-allocate an array for the training set bags of centroids (for speed)
        train_centroids = np.zeros((train.icol(1).size, num_clusters), \
            dtype="float32")
    
        # Transform the training set reviews into bags of centroids
        counter = 0
      
        for review in clean_train_reviews:
            train_centroids[counter] = self.create_bag_of_centroids(review, \
                word_centroid_map)
            counter += 1
    
        # Repeat for test reviews
        test_centroids = np.zeros((test.icol(1).size, num_clusters), \
            dtype="float32")
    
        counter = 0
        for review in clean_test_reviews:
            test_centroids[counter] = self.create_bag_of_centroids(review, \
                word_centroid_map)
            counter += 1
    
        self.trainDataVec = train_centroids
        self.testDataVec = test_centroids
