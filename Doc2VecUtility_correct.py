# -*- coding: utf-8 -*-
'''
Created on 2016年6月1日

@author: sx
'''
#!/usr/bin/env python

import logging
import os.path
import pandas as pd
import numpy as np
try:
    from process.Word2VecUtility import Word2VecUtility
except:
    from Word2VecUtility import Word2VecUtility
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

"""  reviews a list of review ,model  trained already,  """
def getFeatureVecs(reviews, model, num_features):
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    counter = -1
     
    for review in reviews:
        counter += 1
        try:
            reviewFeatureVecs[counter] = np.array(model[review.labels[0]]).reshape((1, num_features))
        except:
            continue
    return reviewFeatureVecs


def getCleanLabeledReviews(reviews,remove_stopwords=False,low_freq_filter=False):
    clean_reviews = []
    wordUtility=Word2VecUtility()
    colnames=reviews.columns.tolist()
    if colnames.__contains__("description")==False:
        for review in reviews['content']:
            sentInreview=wordUtility.tokenreviewtoSentenceEn(review)
            tokenReview=[]
                #将每个句子 处理后 拼接到一个list
            for sent in sentInreview:
                tokenReview+=Word2VecUtility.review_to_wordlist_en(sent, remove_stopwords=remove_stopwords ,low_freq_filter=low_freq_filter)
                # 将这个review  of list 添加到clean_review (list of list [[],[]])  
            clean_reviews.append(tokenReview)
    else:
        for review in reviews["description"]:
            sentInreview=wordUtility.tokenreviewtoSentence(review)
            tokenReview=[]
                #将每个句子 处理后 拼接到一个list
            for sent in sentInreview:
                tokenReview+=Word2VecUtility.review_to_wordlist(sent, remove_stopwords=remove_stopwords ,low_freq_filter=low_freq_filter)
                # 将这个review  of list 添加到clean_review (list of list [[],[]])  
            clean_reviews.append(tokenReview)
            
           
    # 将打乱的 建立 label labeldic：{1:label1,-1:label2} labelCount:{label1:num1,label:num2} 
    labeldic={}
    labelCount={}
    labelized = []
    k=1
    for i in range(len(clean_reviews)):
        if not labeldic.has_key(reviews['label'][i]):
            labeldic[reviews['label'][i]]='label'+str(k)
            
            labelCount['label'+str(k)]=1
            k+=1
        else:
            labelCount[labeldic[reviews['label'][i]]]+=1
        reviewLabel=labeldic[reviews['label'][i]]
        labelized.append(LabeledSentence(clean_reviews[i],[reviewLabel+"_"+str(labelCount[reviewLabel])]))
    return labelized
