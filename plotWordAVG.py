# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:05:47 2017

@author: sx
"""

import os
import re
import pylab as pl
import codecs
import matplotlib.pyplot as plt

plt.figure(1) # 创建图表1
ax1 = plt.subplot(111) # 在图表2中创建子图1
plt.figure(2) # 创建图表2
ax2 = plt.subplot(111) # 在图表2中创建子图1
dic={}
Numkey=0
if os.path.exists("G:/worksp/pythonScrapy/src/process/sst/precison.txt"):
    file = codecs.open('G:/worksp/pythonScrapy/src/process/sst/precison.txt','r','utf-8')
    for line in file.readlines():
        if line.encode('utf-8').strip():
            line=line.encode('utf-8')
            if '相似性阈值' in line:
                Numkey=re.findall("\d*\.\d*",line)[0]
                if not dic.has_key(Numkey):
                    dic[Numkey]={}
                continue
            matchObj = re.match( r'BoWRandomForest精度:', line, re.M|re.I)
            if matchObj:            
                print Numkey,": BoWRandomForest:",line.split(':')[1]
                if dic[Numkey].has_key('BoWRandomForest'):
                    dic[Numkey]['BoWRandomForest'].append(line.split(":")[1].strip())
                else:
                    dic[Numkey]['BoWRandomForest']=[]
                    dic[Numkey]['BoWRandomForest'].append(line.split(":")[1].strip())
            matchObj = re.match( r'BoWLinearSvm精度:', line, re.M|re.I)
            if matchObj:
                print Numkey,": BoWLinearSvm:",line.split(":")[1]
                if dic[Numkey].has_key('BoWLinearSvm'):
                    dic[Numkey]['BoWLinearSvm'].append(line.split(":")[1].strip())
                else:
                    dic[Numkey]['BoWLinearSvm']=[]
                    dic[Numkey]['BoWLinearSvm'].append(line.split(":")[1].strip())
            matchObj = re.match( r'BoWCHMRandomForest精度:', line, re.M|re.I)
            if matchObj:            
                print Numkey,": BoWCHMRandomForest:",line.split(':')[1]
                if dic[Numkey].has_key('BoWCHMRandomForest'):
                    dic[Numkey]['BoWCHMRandomForest'].append(line.split(":")[1].strip())
                else:
                    dic[Numkey]['BoWCHMRandomForest']=[]
                    dic[Numkey]['BoWCHMRandomForest'].append(line.split(":")[1].strip())
            matchObj = re.match( r'BoWCHMLinearSvm精度:', line, re.M|re.I)
            if matchObj:
                print Numkey,": BoWCHMLinearSvm:",line.split(":")[1]
         
                if dic[Numkey].has_key('BoWCHMLinearSvm'):
                    dic[Numkey]['BoWCHMLinearSvm'].append(line.split(":")[1].strip())
                else:
                    dic[Numkey]['BoWCHMLinearSvm']=[]
                    dic[Numkey]['BoWCHMLinearSvm'].append(line.split(":")[1].strip())
            matchObj = re.match( r'BoWCHVRandomForest精度:', line, re.M|re.I)
            if matchObj:            
                print Numkey,": BoWCHVRandomForest:",line.split(':')[1]
      
                if dic[Numkey].has_key('BoWCHVRandomForest'):
                    dic[Numkey]['BoWCHVRandomForest'].append(line.split(":")[1].strip())
                else:
                    dic[Numkey]['BoWCHVRandomForest']=[]
                    dic[Numkey]['BoWCHVRandomForest'].append(line.split(":")[1].strip())
            matchObj = re.match( r'BoWCHVLinearSvm精度:', line, re.M|re.I)
            if matchObj:
                print Numkey,": BoWCHVLinearSvm:",line.split(":")[1]
     
                if dic[Numkey].has_key('BoWCHVLinearSvm'):
                    dic[Numkey]['BoWCHVLinearSvm'].append(line.split(":")[1].strip())
                else:
                    dic[Numkey]['BoWCHVLinearSvm']=[]
                    dic[Numkey]['BoWCHVLinearSvm'].append(line.split(":")[1].strip())
            matchObj = re.match( r'WAVRandomForest精度:', line, re.M|re.I)
            if matchObj:
                print Numkey,": WAVRandomForest:",line.split(":")[1]
     
                if dic[Numkey].has_key('WAVRandomForest'):
                    dic[Numkey]['WAVRandomForest'].append(line.split(":")[1].strip())
                else:
                    dic[Numkey]['WAVRandomForest']=[]
                    dic[Numkey]['WAVRandomForest'].append(line.split(":")[1].strip())
            matchObj = re.match( r'WAVLinearSvm精度:', line, re.M|re.I)
            if matchObj:
                print Numkey,": WAVLinearSvm:",line.split(":")[1]
     
                if dic[Numkey].has_key('WAVLinearSvm'):
                    dic[Numkey]['WAVLinearSvm'].append(line.split(":")[1].strip())
                else:
                    dic[Numkey]['WAVLinearSvm']=[]
                    dic[Numkey]['WAVLinearSvm'].append(line.split(":")[1].strip())
            matchObj = re.match( r'WBCRandomForest精度:', line, re.M|re.I)
            if matchObj:
                print Numkey,": WBCRandomForest:",line.split(":")[1]
     
                if dic[Numkey].has_key('WBCRandomForest'):
                    dic[Numkey]['WBCRandomForest'].append(line.split(":")[1].strip())
                else:
                    dic[Numkey]['WBCRandomForest']=[]
                    dic[Numkey]['WBCRandomForest'].append(line.split(":")[1].strip())
            matchObj = re.match( r'WBCLinearSvm精度:', line, re.M|re.I)
            if matchObj:
                print Numkey,": WBCLinearSvm:",line.split(":")[1]
     
                if dic[Numkey].has_key('WBCLinearSvm'):
                    dic[Numkey]['WBCLinearSvm'].append(line.split(":")[1].strip())
                else:
                    dic[Numkey]['WBCLinearSvm']=[]
                    dic[Numkey]['WBCLinearSvm'].append(line.split(":")[1].strip())
else:
    print 'file not found,please check the filesystem'
    
newDic={}
classNum={'BoWLinearSvm':0,'BoWRandomForest':1,'WAVLinearSvm':2,'WAVRandomForest':3,'WBCLinearSvm':4,'WBCRandomForest':5,'BoWCHMLinearSvm':6,'BoWCHMRandomForest':7,'BoWCHVLinearSvm':8,'BoWCHVRandomForest':9}
avg=[0,0,0,0,0,0,0,0,0,0]
avgNum=[0,0,0,0,0,0,0,0,0,0]
for itemKey in dic:
    
    for eachClassItem in dic[itemKey]:
       
        for inum in dic[itemKey][eachClassItem]:
            
            avg[classNum[eachClassItem]]+=float(inum.strip())
            avgNum[classNum[eachClassItem]]+=1

for i in range(len(avg)):
    avg[i]=avg[i]/avgNum[i]
print avg    
    
"""
newDic={}

for itemKey in dic:
    newDic[itemKey]={}
    for eachClassItem in dic[itemKey]:
        res=0.
        num=0
        for inum in dic[itemKey][eachClassItem]:
            res+=float(inum.strip())
            num+=1
        newDic[itemKey][eachClassItem]=res/num
file.close()
print len(dic)
print dic
yVF=[]
yMF=[]
yF=[]
yVS=[]
yMS=[]
yS=[]
x = ['0.6', '0.65','0.7','0.75', '0.8', '0.85','0.9','0.95']# Make an array of x values
for itemKey in x:
    print itemKey
    for eachClassItem in dic[itemKey]:
        if eachClassItem=='BoWCHVLinearSvm':
            yVS.append(newDic[itemKey][eachClassItem])
        if eachClassItem=="BoWCHMLinearSvm":
            yMS.append(newDic[itemKey][eachClassItem])
        if eachClassItem=="BoWLinearSvm":
            yS.append(newDic[itemKey][eachClassItem])
            
        if eachClassItem=='BoWCHVRandomForest':
            yVF.append(newDic[itemKey][eachClassItem])
        if eachClassItem=="BoWCHMRandomForest":
            yMF.append(newDic[itemKey][eachClassItem])
        if eachClassItem=="BoWRandomForest":
            yF.append(newDic[itemKey][eachClassItem])
 """
#yVsum=[]
#yMsum=[]
#ysum=[]
#for i in range(len(yS)):
#    yVsum.append((yVF[i]+yVS[i])/2)
#    yMsum.append((yMF[i]+yMS[i])/2)
#    ysum.append((yF[i]+yS[i])/2)
"""

plt.sca(ax1)
plt.title(u"分类的准确度Y VS 阈值X",fontproperties='SimHei')
plt.ylabel(u'Y 轴',fontproperties='SimHei')
plt.xlabel(u'X 轴',fontproperties='SimHei')

plt.xlim(0.550,0.950)
plt.ylim(0.675,0.725)
plot1=plt.plot(x, yVF,'ok-')# use pylab to plot x and y
#plot2=pl.plot(x, yVS,'ob--')# use pylab to plot x and y

plt.text(x[4],yVF[4],u"1.词袋模型结合词向量（多个词平均）",fontproperties='SimHei')
plot1=plt.plot(x, yMF,'ok-')# use pylab to plot x and y
#plot2=pl.plot(x, yMF,'og--')# use pylab to plot x and y
plt.text(x[4],yMF[4],u"2.词袋模型结合词向量（最相近词）",fontproperties='SimHei')

plot1=plt.plot(x, yF,'ok-')# use pylab to plot x and y
#plot2=pl.plot(x, yF,'or--')# use pylab to plot x and y
plt.text(x[4],yF[4],u"3.词袋模型",fontproperties='SimHei')
plt.legend([u'1.词袋模型结合词向量（多个词平均）',u'2.词袋模型结合词向量（最相近词）',u'3.词袋模型'],prop={'family':'SimHei','size':15})# make legend


plt.sca(ax2)
plt.title(u'分类的准确度Y VS 阈值X',fontproperties='SimHei')
plt.ylabel(u'Y 轴',fontproperties='SimHei')
plt.xlabel(u'X 轴',fontproperties='SimHei')

plt.xlim(0.550,0.950)
plt.ylim(0.770,0.795)
#plot1=pl.plot(x, yVF,'ob-')# use pylab to plot x and y
plot2=plt.plot(x, yVS,'ob-')# use pylab to plot x and y

#plot1=pl.plot(x, yMF,'og-')# use pylab to plot x and y
plot2=plt.plot(x, yMS,'og--')# use pylab to plot x and y


#plot1=pl.plot(x, yF,'or-')# use pylab to plot x and y
plot2=plt.plot(x, yS,'or-')# use pylab to plot x and y


plt.legend(['BoWVLinearSvm precision', 'BoWMLinearSvm precision','BoWLinearSvm precision'])#,prop={'family':'SimHei','size':15})# make legend

pl.show()# show t

"""