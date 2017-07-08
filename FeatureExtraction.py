#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2016年12月23日

@author: sx
'''


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import _document_frequency

def TestcalulateTfIdf(corpus):
    vec=TfidfVectorizer()
    transformer = TfidfTransformer()
    """counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]]
          """
    corpus= ['This is the first document.',
             'This is the second second document.',
             'And the third one.',
             'Is this the first document?']
    tfidf=vec.fit_transform(corpus)
    tfidf_mat=tfidf.toarray()
    voc,X=vec._count_vocab(corpus,vec.vocabulary_)
    X_mat=X.toarray()
             
    print "TF:",X_mat
    print "idf:",vec.idf_
    print "tfidf:",tfidf_mat
    matrix=np.zeros(X_mat.shape)
    i=-1
    for row in X_mat:
        i+=1
        matrix[i]= row*vec.idf_
    calTfmatrix=np.zeros(matrix.shape)
    i=0
    for row in matrix:
        calTfmatrix[i]=row/np.sqrt(sum(np.square(row)))
        i+=1
    print "calTFidf:",calTfmatrix
def FeatureExtractSimilarCon(modelWord,corpus=None,term_set_fs=None,similarVal=0.6):
    
    if term_set_fs!=None:
        term_dict = dict(zip(term_set_fs, range(len(term_set_fs))))
      
        vectorizer=TfidfVectorizer(vocabulary= term_dict)
    else:
        vectorizer=TfidfVectorizer()
    tfidf=vectorizer.fit_transform(corpus)
    vocabularyCorpus=vectorizer.vocabulary_
     # 为值为0的添加其他值
        
    vocAndIndex_unicodeTostr={}
    for item in vocabularyCorpus:
        vocAndIndex_unicodeTostr[item.encode("utf-8")]=vocabularyCorpus[item]
    vocabularyCorpus=vocAndIndex_unicodeTostr
    
    
    
    simlardic={}#保存相似词列表{word1:[simword]，word2:[simword]},列表里面每个是相似词
    existSimlarlist=[]#保存已经计算的词
    newVoc={}
    i=0
    count=0# 计数总共减少了多少特征词
    for item in sorted(vocabularyCorpus.items(),key=lambda d:d[1]):#对于词库表中的每个词
        word=item[0]
        if  word not in existSimlarlist:#当前词没有计算或者没有通过计算被得到
            newVoc[word]=i
            i+=1
            if modelWord.vocab.has_key(word):#当前词存在词向量
                simlarlis=modelWord.most_similar(word)#计算相似词list
                simlist=[]
                for sim,simVal in simlarlis:
                    if simVal>similarVal and sim in vocabularyCorpus:#相似性达到阈值，并且在原词表中
                        existSimlarlist.append(sim)#sim词也相当于计算了
                        count+=1
                        simlist.append(sim)
                    else:
                        break
                if len(simlist)>0:#如果计算得到simlist【】
                    simlardic[word]=simlist#把word:simlist 键值对加入到simlarlist中
        existSimlarlist.append(word)#把当前词保存到已计算词表中
    
    print "合并相似词："
    tfidf_mat=tfidf.toarray()
    # voc 词库 Xold 词频矩阵
    voc,Xold=vectorizer._count_vocab(corpus,True)
    X_mat=Xold.toarray()
    Xnew_mat=np.zeros((len(X_mat),(len(newVoc))))
    print Xnew_mat.shape
    f=open("./sst/FeatureConbineOrNotprecison.txt",'a')
    f.write("维度：")
    f.write(str(len(voc)))
    f.write("新维度：")
    f.write(str(len(newVoc)))
    f.write('\n')
    f.close()  
    #构造新的Xnew_mat矩阵
    for items in sorted(newVoc.items(),key=lambda d:d[1]):
        word=items[0]
        indexNew=items[1]
        index=0
        #print "word:",word," indexNew:",indexNew
        if word not in simlardic:
            try:
                index=voc[word.decode('utf-8')]
                #print "index:",index
            except:
                print "key not found"
            Xnew_mat[:,indexNew]=X_mat[:,index]
        else:
            index=voc[word.decode('utf-8')]
            Xnew_mat[:,indexNew]=X_mat[:,index]
            for w in simlardic[word]:
                VocsimIndex=voc[w.decode("utf-8")]
                Xnew_mat[:,indexNew]+=X_mat[:,VocsimIndex]
           
    print "TFnew:",Xnew_mat
    
#    for word in newVoc:
#        if word in simlardic:#tf ，df 值都需要加和，list中的值，重新计算idf----tf*idf
#            feature_df=_document_frequency(Xnew_mat)
#            df_sum=0#相似词的df值取或
#            for w in simlardic[word]:
#                vocIndex=voc[w]
#                df_sum+=feature_df[vocIndex]
#                
    # else:#不变
    transformer = TfidfTransformer()
    tfidfnew_mat=transformer.fit_transform(Xnew_mat).toarray()
    return tfidfnew_mat
TestcalulateTfIdf("")