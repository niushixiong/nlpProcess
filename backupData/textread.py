#coding:utf-8
"""
Created on Fri Feb 26 15:18:19 2016

@author: shixiong
和textparserReader 一个功能，此脚本以旧版本，新功能见textParserReader
"""
import nltk
import os
import xml.sax
import re
import numpy  
import pandas as pd
class PatentHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        self.title = ""
        self.keyword = ""
        self.description = ""
        self.id = ""
   
    def startElement(self, tag, attributes):
        self.CurrentData = tag
      

    def endElement(self, tag):
        
        if self.CurrentData == "title":
            print " "
            print self.title
         
         
         
        elif self.CurrentData == "keyword":
            
            print " "
            print self.keyword
            
        elif self.CurrentData == "description":
            
        
            print " "            
            print self.description   
        
        self.CurrentData=""
   
  # 内容事件处理
    def characters(self, content):
        
        if self.CurrentData == "title":
            self.title = content
           
         
        elif self.CurrentData == "keyword":
            self.keyword = content
            
            
        elif self.CurrentData == "description":
            self.description = content
        
            
       # elif self.CurrentData == "id":
        #    print "id:"
         #   self.CurrentData=content
          #  print self.id



alllist=[]
allFileNum = 0  
filepathlist=[]
def getPathfile(level, path):  
    global allFileNum
    global filepathlist
    ''''' 
    打印一个目录下的所有文件夹和文件 
    '''  
    # 所有文件夹，第一个字段是次目录的级别  
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
            getPathfile((int(dirList[0]) + 1), path + '/' + dl)  
    for fl in fileList:  
        # 打印文件  
        print '-' * (int(dirList[0])), fl  
     #   self.parse()
        # 随便计算一下有多少个文件  
        allFileNum = allFileNum + 1  
        filepathlist.append(path+'/'+fl)
        
    return filepathlist
    
   
if __name__ == "__main__":
   # 创建一个 XMLReader
    parser = xml.sax.make_parser()
  # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
  
   # 重写 PatentHandler
    Handler = PatentHandler()
    parser.setContentHandler(Handler)
    
    # 参考目录，当前数据文件夹根目录    
    pwd="../Desktop"
    #解析单个文件
    
 #解析目录所有文件   
    filels=getPathfile(1, pwd+'/dataset_617146')  

    print '总文件数 =', allFileNum
    i=0
    for f in filels:
        pattern=re.compile(r'.*?xml$')
        match =pattern.match(f)
#        if(re.findall)
#        parser.parse(f)
        if match:
            i=i+1
            #print f
            parser.parse(f)
    
            alllist.append([Handler.title.encode('utf-8'),Handler.keyword.encode('utf-8'),Handler.description.encode('utf-8')])
            
            #parser.parse(f) 
   
    print i
    for i in range(10):
        print type(alllist[i][0])
        print alllist[i][1]
        #print alllist[i][1].encode('utf-8')
    
    data=pd.DataFrame(alllist,columns=['title','keyword','description'])
    