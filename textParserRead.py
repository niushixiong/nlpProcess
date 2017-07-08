#coding:utf-8
"""
Created on Fri Feb 26 15:18:19 2016
edit by date： 5.31 
scrapy331 里面有 def getAbContent(self,content):函数 处理单个文件解析工作，和testAbractCont 一致
@author: shixiong
"""
import nltk
import os
import xml.sax
import re
import numpy
from bs4 import BeautifulSoup  
import pandas as pd

""" 解析html文档，得到title ipcNum idNum description"""
class PatentHandler():
    def __init__(self):
        self.title = ""
        self.description = ""
        self.IPCNum=""
        self.idNum = "" 
    # 增加返回值    
    def parseHtml(self,path):
        f=open(path)
        htmltext=f.read()
        soup=BeautifulSoup(htmltext)
        # 摘要 解析
        descont=soup.select(".content")
        if len(descont)<1:
            print path
            print "此文件没有摘要"
            return 0
        #print descont[0].getText()
        self.description=descont[0].getText().strip()
        # 标题解析
        titlecont=soup.select(".fmbt")
        if titlecont==None:
            print "此没有标题，文件出错"
            return 0
        try:
            self.title=titlecont[0].string.split("--")[1].strip()
        except Exception:
            return 0
        # 详细信息解析
        divcont=soup.select("#abstractItemList")
        if divcont==None:
            print "详细信息空，文件错误"
            return 0
        trs=divcont[0].findAll("tr",recursive=True)
        tds=divcont[0].findAll("td",recursive=True)
        tdkey=[]
        tdvalue=[]
        for i in range(len(trs)):
            tdcont=trs[i].contents
            for t in range(len(tdcont)):
                str=tdcont[t].string.strip()
            """ 存在空白tdkey tdvalue为 tdcont[1] tdcont[3]"""
            tdkey.append(tdcont[1].string.strip())
            tdvalue.append(tdcont[3].string.strip())
        dicts={}
        for i in range(len(tdkey)):
            dicts[tdkey[i]]=tdvalue[i]
        dicts[u'发明名称']=self.title
        dicts[u'摘要']=self.description
        for d,x in dicts.items():
            if type(d)=="str":
                d=d.decode("utf-8")
            if type(x)=="str":
                x=x.decode("utf-8")
        if dicts.has_key(u'IPC分类号'):
            self.IPCNum=dicts[u'IPC分类号']
        id=soup.find("span",class_="current").string.strip()
        if id==None:
            print "没有find id，文件出错"
            return 0
        idNum=id.split("[")[0].strip()
        self.idNum=idNum
        f.close()
        return 1

# 读取文件目录里面的所有文件并解析，为DataFrame类型
class textParserRead():
    def __init__(self):
        self.alllist=[]
        self.allFileNum = 0  
        self.filepathlist=[]
        self.pwd="C:\Users\shixiong\Desktop"
        self.data=0
    def getPathfile(self,level, path):  
        dirList = []  
        # 所有文件  
        fileList = []  
        # 返回一个列表，其中包含在目录条目的名称(google翻译)  
        files = os.listdir(path)  
        # 先添加目录级别  
        dirList.append(str(level))  
        for f in files:  
            if(os.path.isdir(path + '/' + f)):  
                # 排除隐藏文件夹。因为隐藏文件夹过多  
                if(f[0] == '.'):  
                    pass  
                else:  
                    # 添加非隐藏文件夹  
                    dirList.append(f)  
            if(os.path.isfile(path + '/' + f)):  
                # 添加文件  
                fileList.append(f)  
        # 当一个标志使用，文件夹列表第一个级别不打印  
        i_dl = 0  
        for dl in dirList:  
            if(i_dl == 0):  
                i_dl = i_dl + 1  
            else:  
                # 打印至控制台，不是第一个的目录  
                print '-' * (int(dirList[0])), dl  
                # 打印目录下的所有文件夹和文件，目录级别+1  
                pwd=path+'/'+dl
                self.getPathfile((int(dirList[0]) + 1), pwd)  
        for fl in fileList:  
            # 打印文件  
            print '-' * (int(dirList[0])), fl  
            # 随便计算一下有多少个文件  
            self.allFileNum = self.allFileNum + 1  
            self.filepathlist.append(path+'/'+fl)
            
        return self.filepathlist
    def getFramdata(self):

        parser = PatentHandler()   
        # 参考目录，当前数据文件夹根目录    
        #pwd="../Desktop"
        #解析单个文件
        #解析目录所有文件 
        #tpr=textParserRead()
        path=self.pwd+"/download"
        filels=self.getPathfile(1,path)  
    
        print '总文件数 =', self.allFileNum
        i=0
        for f in filels:
            pattern=re.compile(r'.*?txt$')
            match =pattern.match(f)
            if match:
        #   if i<5:
                i=i+1
                #print f
                res=parser.parseHtml(f)
                if res==1:
                    # alllist 中的 每个element 由 unicode to str
                    self.alllist.append([parser.title.encode('utf-8'),parser.idNum.encode('utf-8'),parser.IPCNum.encode('utf-8'),parser.description.encode('utf-8')])
        self.data=pd.DataFrame(self.alllist,columns=['title','idNum','IPCNum','description'])
        return self.data    
   
if __name__ == "__main__":
    ptr=textParserRead()
    ptr.getFramdata()
    data=ptr.data