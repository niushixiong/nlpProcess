#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import shuffle
import numpy as np
try:
    from textParserRead import textParserRead
    from textProcess import textProcess
    from dA import dA
    from autoencoder import DenoisingAutoencoder
except:
    from process.textParserRead import textParserRead
    from process.textProcess import textProcess
    from process.dA import dA
    from process.autoencoder import DenoisingAutoencoder
from collections import namedtuple 
#from emd import emd                         
from math import  sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from numpy.dual import svd
import nltk

class Word2VecUtility():
    """Word2VecUtility is a utility class for processing raw HTML text into segments for further learning"""
   
    #设置分句的标志符号；可以根据实际需要进行修改  
    cutlist ="。！？.!?".decode("utf-8")
    """ review 为str类型 decode("utf-8") 为unicode"""
    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False,low_freq_filter = False ,english=False):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #  
        textprocess=textProcess()
        #words 为list 每一项 str：词组
        words=textprocess.tokenProcess(review,remove_stopwords=remove_stopwords,low_freq_filter = low_freq_filter,english=english)
      
        return(words)

        
    @staticmethod
    def review_to_wordlist_en(review,remove_stopwords=False,low_freq_filter = False):
        textprocess=textProcess()
        words=textprocess.tokenProcessEn(review,remove_stopwords=remove_stopwords,low_freq_filter = low_freq_filter)
        return words
    # Define a function to split a review into parsed sentences  ，英文分句
  
    def review_to_sentences(self, reviews, remove_stopwords=False,low_freq_filter=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')  
        raw_sentences = sent_tokenizer.tokenize(reviews)  
        """"raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        """
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( Word2VecUtility.review_to_wordlist_en( raw_sentence, \
                  remove_stopwords=remove_stopwords ,low_freq_filter=low_freq_filter))
       
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
    #以下为调用上述函数实现从文本文件中读取内容并进行分句。
   
    def contents_to_sentences(self,reviews,remove_stopwords=False,low_freq_filter=False):
        
        sentences=[]
          
        #l = self.Cut(list(self.cutlist),list(review))     
        l = self.Cut(list(self.cutlist),list(reviews.decode('utf-8')))       
        for line in l:    
            if line.strip() !="":        
                li = line.strip().split()  
                # 把li（list）里面的一个句子赋予sentence
                for sentence in li:
                    #print "***************sentence*************************"  
                    #print sentence
                    sentences.append(Word2VecUtility.review_to_wordlist(sentence, remove_stopwords=remove_stopwords,low_freq_filter=low_freq_filter))
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists           
        return sentences                 
    # Define a function to split a DataFrame for training and test
    @staticmethod
    def split_train_test( df, test_portion=0.3 ):
        # create random list of indices
        """ 原来的shuffle函数
        N = len(df)
        
        l = range(N)
        shuffle(l)
 
        # get splitting indicies
        trainLen = int(N*(1-test_portion))
 
        # get training and test sets
        train = df.ix[l[:trainLen]]
        test = df.ix[l[trainLen:]]
        """
        N = len(df)
        ll=df.index.tolist()
        
        shuffle(ll)
 
        # get splitting indicies
        trainLen = int(N*(1-test_portion))
 
        # get training and test sets
        train = df.ix[ll[:trainLen]]
        test = df.ix[ll[trainLen:]]
        
        return train, test
    
 
  
    #检查某字符是否分句标志符号的函数；如果是，返回True，否则返回False  
    def FindToken(self,cutlist, char):  
        if char in cutlist:  
            return True  
        else:  
            return False  
       
    #进行分句的核心函数      
    def Cut(self,cutlist,lines):          #参数1：引用分句标志符；参数2：被分句的文本，为一行中文字符  
        l = []         #句子列表，用于存储单个分句成功后的整句内容，为函数的返回值  
        line = []    #临时列表，用于存储捕获到分句标志符之前的每个字符，一旦发现分句符号后，就会将其内容全部赋给l，然后就会被清空  
              
        for i in lines:         #对函数参数2中的每一字符逐个进行检查 （本函数中，如果将if和else对换一下位置，会更好懂）  
            if self.FindToken(cutlist,i):       #如果当前字符是分句符号  
                line.append(i)          #将此字符放入临时列表中  
                l.append(''.join(line))   #并把当前临时列表的内容加入到句子列表中  
                line = []  #将符号列表清空，以便下次分句使用  
            else:         #如果当前字符不是分句符号，则将该字符直接放入临时列表中  
                line.append(i)       
        return l  
    
   
    """
     #每段文字建立向量 num_features 选择的特征向量长度，model 为已经训练的模型，words为当前文字段
      get a sentence vec by add all of the word of the sentence then divide by the nums of the words
      the word is vec trained by word2vec得到句子的向量形式，把句子的所有单词相加，然后除以单词数，单词的向量形式通过word2vec得到 
    """  
    def makeFeatureVec(self,words, model, num_features,weight=None):
        featureVec = np.zeros((num_features,),dtype="float32")
        nwords = 0
    
        index2word_set = set(model.index2word)
        
        for word in words:
            if word in index2word_set:
                #print model[word]
                #print weight[nwords]
                featureVec = np.add(featureVec,model[word]*weight[nwords])
                nwords = nwords + 1
        if nwords != 0 and weight==None:
            featureVec /= nwords
    
        return featureVec
    
    """ text中的word拼接 :wordi,wordi+1 拼接时有个重叠率 Num_feature：结果生成的向量长度，model中的word有个维度"""
    # words 现在保证数量是一致的，如果不同，如何处理
    def makeLinkWords(self,overlappingRate,words,model,wordfeatureNum):
        # 计算链接所有单词后的向量维度
        nwords = len(words)
        # 每个单词的应取得维度
        realwordfeatureNum=wordfeatureNum*(1-overlappingRate)
        overlappingPart=wordfeatureNum.overlappingRate
        # 文档维度
        num_features=realwordfeatureNum*nwords
        featureVec = np.zeros((num_features,),dtype="float32") 
        # 计算链接后的向量
        index2word_set = set(model.index2word)
        flag=True
        for word in words.split(" "):
            if word in index2word_set:
                if flag==False:
                    #上一个词未在词库（这种情况理论不存在，让他的菲重叠部分为1）
                    featureVec=np.hstack((featureVec[:-overlappingPart],model[word]))
                    continue
                #前后重叠部分求和
                overLap=featureVec[-overlappingPart:]+model[word][:overlappingPart]
                # 连接重叠部分
                featureVec=np.hstack((featureVec[:-overlappingPart],overLap))
                #连接后部分  
                featureVec = np.hstack((featureVec,model[word][overlappingPart:]))
                flag=True
            else: 
                #当前词不存在词库（理论上不存在），让他为【1】
                featureVec=np.hstack((featureVec,np.ones(wordfeatureNum)[overlappingPart:]))
                flag=False
                
        return featureVec
    # 计算所有文本的平均长度
    def getReviewsAvgLen(self,reviews):
        revlens=0
        lenreviews=len(reviews)
        for review in reviews:
            revlens=revlens+len(review)
        return revlens/lenreviews
    # 插值函数： 在向量为featureVec 中插入num*(model of feature) featureWith:词数目 featureVec 原向量，返回处理后的向量
    def myinterpolate(self,featureVec,featureWidth,num):
        #间隔，每间隔wordinterval个词插入一个词
        Wordinterval=(featureWidth-1)/num
        #平均每个词维度
        wordDim=featureVec/featureWidth
        #依次计算每个插入位置
        i=Wordinterval
        count=0
        while i<featureWidth:
            # 计算插入位置的值，每次计算 featureVec长度都增加WordDim
            Interfeat=(featureVec[wordDim*(i-1)+count*wordDim:wordDim*i+count*wordDim]+featureVec[wordDim*i+count*wordDim:wordDim*(i+1)+count*wordDim])/2
            #插入位置前的和插入的拼接
            tempFore=np.hstack((featureVec[:wordDim*i+count*wordDim],Interfeat))
            #合并插入位置后的
            featureVec=np.hstack((tempFore,featureVec[wordDim*i+wordDim*count:])) 
            i=i+Wordinterval
            count=count+1   
        #返回插入num个词后的featureVec    
        return featureVec
        
    #采样函数，在向量为featureVec 中去掉num ,从后部分去掉
    def mysampling(self,featureVec,featureWidth,num):
        Wordinterval=featureWidth/(num+1)
        wordDim=featureVec/featureWidth
        count=0
        i=1
        while(num>0):
    
            featureVec=np.hstack((featureVec[:(i*Wordinterval-1-count)*wordDim],featureVec[(i*Wordinterval-count)*wordDim:]))
            num=num-1
            count=count+1
            i=i+1
        return featureVec
    """
       reviews: list of sentense( list of word)
       model : word2vec
       num_features: features
       return: featurevecs (text vecs (a list of sentence vec))
        #得到所有的文字的向量矩阵
    """
    def getLinkWordFeatureVecs(self,reviews, model,wordfeatureNum,overlappingRate):
        reviewAvgLen=self.getReviewsAvgLen(reviews)
        num_features=wordfeatureNum*(1-overlappingRate)*reviewAvgLen
        reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
        counter = 0
        for review in reviews:
            featureVec=self.makeLinkWords(overlappingRate,review,model,wordfeatureNum)
            if len(review)<reviewAvgLen:
                num=reviewAvgLen-len(review) 
                updatefeatureVec=self.myinterpolate(featureVec,len(review),num)
            elif len(review)>reviewAvgLen:
                num=len(review)-reviewAvgLen
                updatefeatureVec=self.mysampling(featureVec,len(review),num)
            else:
                updatefeatureVec=featureVec
            reviewFeatureVecs[counter] = updatefeatureVec
            counter = counter + 1
        return reviewFeatureVecs

    """ 得到权重值   clean_reviews 一些短文本，对应 getAllFeatureVecs的reviews 这两个函数的reviews参数相同
        model word2vec 训练得到的模型. 
        return : 所有文本的每个单词对应的权重值，和getAlllFeaturevecs 一起使用"""
    def getFeatureWeight(self,clean_reviews,model):
        vectorizer = TfidfVectorizer(min_df=1)
        data_features = vectorizer.fit_transform(clean_reviews)
        # Numpy arrays are easy to work with, so convert the result to an
        # array
        TfidifvectorArray = data_features.toarray()
        #权重 list 每一项是 一个list 里面代表每一行的词权重
        reviewWeights= []
        # 所有的特征词
        Vocabulary=data_features.vocabulary_
        # model 单词 索引 集合
        index2word_set = set(model.index2word)
        # 第几项文本
        reviewLine=0
        for words in clean_reviews: 
            wordweight=[]
            for word in words.split(" "):
                if word in index2word_set:
                    #获得 当前单词所在的索引列
                    wordIndex=Vocabulary.get(word)
                    if wordIndex==None:
                        wordweight.append(0)
                    else:
                        wordweight.append(TfidifvectorArray[reviewLine][wordIndex])
            # 归一化，权重和为1
            wordoneweight=[]
            for i in range(len(wordweight)):
                wordoneweight.append(wordweight[i]/sum(wordweight))
            reviewWeights.append(wordoneweight)
            reviewLine=reviewLine+1
        return reviewWeights
        
    """
     #每段文字建立向量 num_features 选择的特征向量长度，model 为已经训练的模型，words为当前文字段
      the word is vec trained by word2vec得到句子的向量形式，把句子的所有单词拼接起来，单词的向量形式通过word2vec得到 
      example:
      Feature = namedtuple("Feature", ["x", "y", "z"])
      features1 = [Feature(100, 40, 22), Feature(211, 20, 2),
                 Feature(32, 190, 150), Feature(2, 100, 100)]
    """  
    def makeFeatureVesAppend(self,words,model,num_features):
        featureColumnName=[]
        for i in range(num_features):
            featureColumnName.append("feature"+str(i))
        FeatureVec = namedtuple("Feature", featureColumnName)
        nwords = 0
        index2word_set = set(model.index2word)
        for word in words.split(" "):
            if word in index2word_set:
                nwords = nwords + 1
                FeatureVec.append(model[word])
    
        if nwords != 0:
            print "文本的宽度：",nwords
        return FeatureVec
    #获得reviews的所有向量
    def getAllFeatureVecs(self,reviews,model,num_features):
        #reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
        reviewFeatureVecs=[]
        counter = 0
        for review in reviews:
            #reviewFeatureVecs[counter] = self.makeFeatureVesAppend(review, model, num_features)
            reviewFeatureVecs.append(self.makeFeatureVesAppend(review, model, num_features))
            counter = counter + 1
        return reviewFeatureVecs
    # 得到文本矩阵形式，然后统一维度处理
    def getAllWordsMatrix(self,reviews,model,num_features,line_avg_words,short_words,policy='DA'):
        reviewFeaturesVecs=[]
        #da = DenoisingAutoencoder(n_hidden=line_avg_words) 
        i=0
        for review in reviews:
            i+=1
            nwords = 0
            index2word_set = set(model.index2word)
            FeatureVec=[]
            for word in review.split(" "):
                #print word
                if word in index2word_set:
                    nwords = nwords + 1
                    FeatureVec.append(model[word])
                else:
                    print review,":",word
            #if nwords != 0:
                # print "文本的宽度：",nwords
                #continue
            if nwords==0:
                f=open("./Notcal.txt",'a')
                f.write(str(i))
                f.write(review.split(" "))
                f.write('\n')
                f.close()
                continue
            
            if policy=='DA':
                print "line_avg_words:" ,line_avg_words  
                reviewMatrix=np.array(FeatureVec).T
                print "reviewMatrix shape:",reviewMatrix.shape
                reviewMatrix_chd=self.processMatrixDA(reviewMatrix,line_avg_words)
            else:
                print "short_words:" ,short_words  
                reviewMatrix=np.array(FeatureVec).T
                print "reviewMatrix shape:",reviewMatrix.shape
                reviewMatrix_chd=self.processMatrixSVD(reviewMatrix, short_words-1)
            #da.fit(reviewMatrix)
            # reviewMatrix_chd=da.transform_latent_representation(reviewMatrix)
            
            reviewFeaturesVecs.append(reviewMatrix_chd.T)
            
        
        return np.array(reviewFeaturesVecs)
    #process matrix from to 
    def processMatrixSVD(self,X,NewDim):
        svd=TruncatedSVD(n_components=NewDim, n_iter=10, random_state=42)
        svd.fit(X)
        X1=svd.transform(X)
        return X1

    def processMatrixDA(self,X,NewDim):
        da = DenoisingAutoencoder(n_hidden=NewDim)
        da.fit(X)
        X1=da.transform_latent_representation(X)
        return X1
    #距离定义
    def distance(self,f1, f2):
        return sqrt( (f1.x - f2.x)**2  + (f1.y - f2.y)**2 + (f1.z - f2.z)**2 )
    """#距离矩阵 
    trainViewFeatureVecs训练特征矩阵   
    testViewFeatureVecs测试特征矩阵    
    trainWeightVecs,训练权重矩阵
    testWeightVecs 测试权重矩阵
    return: disMatric 距离矩阵
    """
    def getDistanceMatric(self,trainViewFeatureVecs,testViewFeatureVecs,trainWeightVecs,testWeightVecs):
        disMatric = np.zeros((len(testViewFeatureVecs),(len(trainViewFeatureVecs))),dtype="float32")
        for i in range(len(testViewFeatureVecs)):
            for j in range(len(trainViewFeatureVecs)):
                disMatric[i][j]=emd((testViewFeatureVecs[i], testWeightVecs[i]), (trainViewFeatureVecs[j], trainWeightVecs[j]),self.distance)
        return disMatric
    """
       reviews: list of sentense( list of word)
       model : word2vec
       num_features: features
       return: featurevecs (text vecs (a list of sentence vec))
        #得到所有的文字的向量矩阵
    """
    def getAvgFeatureVecs(self,reviews, model, num_features):
        reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
        counter = 0
        countervector=TfidfVectorizer()
        raw_reviews=[]
        for line in reviews:
            raw_reviews.append(" ".join(line))
        vec=countervector.fit_transform(raw_reviews)

        CouVacalubarys=countervector.vocabulary_
        Featurewords=countervector.get_feature_names()#获取词袋模型中的所有词语
        vecarray=vec.toarray()
        weight=[]
        for review in reviews:
            # 增加权重设置,第一句话的权重大些 乘以1.2，其他的为counterVectorizer训练得到
            for word in review:
                #bug "word为str 必须解码为utf-8"
                j=CouVacalubarys.get(word.decode("utf-8"))
                if(j==None):
                    weight.append(0.000001)  
                else:     
                    weight.append(vecarray[counter][j].tolist())
                #print vecarray[counter][j].tolist()
            reviewFeatureVecs[counter] = self.makeFeatureVec(review, model, num_features,weight)
            counter = counter + 1
        return reviewFeatureVecs
    
    # 把文档分成一个个句子
    def tokenreviewtoSentenceEn(self,raw):
        #分割成句子  
        sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')  
        sents = sent_tokenizer.tokenize(raw)  
        return  sents  
    #把文档分成一个个句子
    def tokenreviewtoSentence(self,raw):
        sents = self.Cut(list(self.cutlist),list(raw.decode('utf-8')))       
        return sents
    # 不加分句处理，直接对文本预处理（去标点，去停顿词等），返回二维list[[]] 旧版本,new ：更新分句然后处理

    def getCleanReviews(self,reviews,remove_stopwords=False,low_freq_filter=False):
        clean_reviews = []
        # review 为str reviews为dataframe 里面的项为str类型 处理的是中文patent 数据集
        colnames=reviews.columns.tolist()
        if colnames.__contains__("description")==True:
        
            for review in reviews["description"]:
                sentInreview=self.tokenreviewtoSentence(review)
                tokenReview=[]
                #将每个句子 处理后 拼接到一个list
                for sent in sentInreview:
                    tokenReview+=Word2VecUtility.review_to_wordlist(sent, remove_stopwords=remove_stopwords ,low_freq_filter=low_freq_filter)
                # 将这个review  of list 添加到clean_review (list of list [[],[]])  
                clean_reviews.append(tokenReview)
        # review 处理的是 Stanford sst 数据集
        else:
            for review in reviews["content"]:
                sentInreview=self.tokenreviewtoSentenceEn(review)
                tokenReview=[]
                #将每个句子 处理后 拼接到一个list
                for sent in sentInreview:
                    tokenReview+=Word2VecUtility.review_to_wordlist_en(sent, remove_stopwords=remove_stopwords ,low_freq_filter=low_freq_filter)
                # 将这个review  of list 添加到clean_review (list of list [[],[]])  
                clean_reviews.append(tokenReview)
                
        return clean_reviews
    # 不加分句处理，直接对文本预处理（去标点，去停顿词等）,返回list[] 旧版本,new：更新分句然后处理
   
    def getCleanReviewsStr(self,reviews,remove_stopwords=False ,low_freq_filter=False):
        clean_reviews = []
        #  review 为str reviews为dataframe 里面的项为str类型 处理的是中文patent 数据集
        colnames=reviews.columns.tolist()
        if colnames.__contains__("description")==True:
        
            for review in reviews["description"]:
                sentInreview=self.tokenreviewtoSentence(review)
                tokenReview=[]
                #将每个句子 处理后 拼接到一个list
                for sent in sentInreview:
                    tokenReview+=Word2VecUtility.review_to_wordlist(sent, remove_stopwords=remove_stopwords ,low_freq_filter=low_freq_filter)
                # 将这个review  of list 添加到clean_review (list of list [[],[]])  
                clean_reviews.append(" ".join(tokenReview))
              
        else:
            # review 处理的是 Stanford sst 数据集
            for review in reviews["content"]:
                sentInreview=self.tokenreviewtoSentenceEn(review)
                tokenReview=[]
                #将每个句子 处理后 拼接到一个list
                for sent in sentInreview:
                    tokenReview+=Word2VecUtility.review_to_wordlist_en(sent, remove_stopwords=remove_stopwords ,low_freq_filter=low_freq_filter)
                # 将这个review  of list 添加到clean_review (list of list [[],[]])  
                clean_reviews.append(" ".join(tokenReview))
              
        return clean_reviews
    
if __name__=="__main__":
    word2vecUtility=Word2VecUtility()
   # tpr=textParserRead()
  #  framedata=tpr.getFramdata()
    
    #for lis  in tpr.data["description"]:
#        print type(lis)
#        print lis
#        t.tokenProcess(lis)
   # lis=tpr.data['description'][1]
    lis1="一种头枕颈枕靠背合一的座椅，可用于飞机、火车、长途客车、中巴车和剧院、办公室、家庭、公共场所，它是在座椅靠背前面上部和头枕体的前面下部增加一个颈枕体，将头枕体、颈枕体和座椅靠背三者合为一体，构成一个头枕、颈枕、座椅靠背组合体。头枕、颈枕、座椅靠背组合体一同随座椅靠背调整转动，对乘坐人的头部、颈部和背部同时提供支撑和保护，防止乘坐人颈部疲劳，当乘坐人的飞机、火车、长途客车、中巴车加速向前运动时，或发生被追尾碰撞的交通事故时，乘坐人最易受伤的颈部受到有效地支撑和保护，防止颈部受伤。"
    lis="At the end , when the now computerized Yoda finally reveals his martial artistry , the film ascends to a kinetic life so teeming that even cranky adults may rediscover the quivering kid inside ."
  
    print "contents to sentences start:"
    sens=word2vecUtility.contents_to_sentences(lis1)
    sens1=word2vecUtility.review_to_sentences(lis)
    print "contents to sentences ends"
    print  " ".join(sens[1])

    