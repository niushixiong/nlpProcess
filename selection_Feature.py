# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 09:49:26 2016

@author: shixiong
"""
import os
import sys

import numpy as np
# 词表sorted
def get_term_dict(doc_terms_list):
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            term_set_dict[term] = 1
    term_set_list = sorted(term_set_dict.keys())       #term set 排序后，按照索引做出字典
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_set_dict
#类别，自增id
def get_class_dict(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    return  class_dict
#
#def stats_term_df(doc_terms_list, term_dict):
#    term_df_dict = {}.fromkeys(term_dict.keys(), 0)
#    for term in term_set:
#        for doc_terms in doc_terms_list:
#            if term in doc_terms_list:
#                term_df_dict[term] +=1                
#    return term_df_dictgi
#对类别计数，每个类别出现一次，加1，返回一个list 每个类别的个数
def stats_class_df(doc_class_list, class_dict):
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list
#获得词表中的词对于类别的贡献值
def stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict):
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float32)
    for k in range(len(doc_class_list)):#对于每个doc 的索引k
        class_index = class_dict[doc_class_list[k]]# doc_class_list[k] 为对应的类别，class——dict 类别对于的index
        doc_terms = doc_terms_list[k]# 对于list of list【【】】中的第K个文章
        for term in set(doc_terms):#对于每个样本集合中的每个词
            term_index = term_dict[term]#在词表中找到这个词的index
            term_class_df_mat[term_index][class_index] +=1#每个词对于类别的贡献值，这个词在每个类别中出现一次加1
    return  term_class_df_mat
        
def feature_selection_mi(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    class_set_size = len(class_df_list)
    
    term_score_mat = np.log(((A+1.0)*N) / ((A+C) * (A+B+class_set_size)))# + * 矩阵元素操作
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[: : -1]# [::-1] start end step 倒序  argsort函数返回的是数组值从小到大的索引值
    term_set_fs = [term_set[index] for index in sorted_term_score_index] #将倒序索引对应的词汇表取出来
    
    return term_set_fs

def feature_selection_ig(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])# A 1.00	5.00和B 5.00 1.00类别贡献值互换(每两项的和减去这两项，矩阵相减)
    C = np.tile(class_df_list, (A.shape[0], 1)) - A# 没出现 词汇的 当前类别的 文档数目
    N = sum(class_df_list)
    D = N - A - B - C
    term_df_array = np.sum(A, axis = 1)
    class_set_size = len(class_df_list)
    
    p_t = term_df_array / N
    p_not_t = 1 - p_t
    p_c_t_mat =  (A + 1) / (A + B + class_set_size)# 矩阵 和A的维度相同
    p_c_not_t_mat = (C+1) / (C + D + class_set_size)
    p_c_t = np.sum(p_c_t_mat  *  np.log(p_c_t_mat), axis =1)
    p_c_not_t = np.sum(p_c_not_t_mat *  np.log(p_c_not_t_mat), axis =1)
    
    term_score_array = p_t * p_c_t + p_not_t * p_c_not_t
    sorted_term_score_index = term_score_array.argsort()[: : -1]#排序逆序把索引组成list
    term_set_fs = [term_set[index] for index in sorted_term_score_index]    
    
    return term_set_fs

def feature_selection_wllr(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C_Total = np.tile(class_df_list, (A.shape[0], 1))
    N = sum(class_df_list)
    C_Total_Not = N - C_Total
    term_set_size = len(term_set)
    
    p_t_c = (A + 1E-6) / (C_Total + 1E-6 * term_set_size)
    p_t_not_c = (B +  1E-6) / (C_Total_Not + 1E-6 * term_set_size)
    term_score_mat = p_t_c  * np.log(p_t_c / p_t_not_c)
    
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    
    print term_set_fs[:10]
    return term_set_fs

def feature_selection(doc_terms_list, doc_class_list, fs_method):# doc_terms_list : list of list[["a","word',"is"],["cut","by","system"]] doc_class_list [labels]
    #如果是专利文本，类别取前三位
   
    class_dict = get_class_dict(doc_class_list)#类别dict
    term_dict = get_term_dict(doc_terms_list)#词表dict
    class_df_list = stats_class_df(doc_class_list, class_dict)##对类别计数，每个类别出现一次，加1，返回一个list 每个类别的个数
    term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict)##获得词表中的词对于类别的贡献值
    term_set = [term[0] for term in sorted(term_dict.items(), key = lambda x : x[1])]#按照term【1】排序，构成term【0】组成的list,str类型每个元素
    term_set_fs = []
    
    if fs_method == 'MI':
        term_set_fs = feature_selection_mi(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'IG':
        term_set_fs = feature_selection_ig(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'WLLR':
        term_set_fs = feature_selection_wllr(class_df_list, term_set, term_class_df_mat)
        
    return term_set_fs