# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:59:31 2017

@author: sx
"""


import os
import re
import pylab as pl
import codecs
import matplotlib.pyplot as plt

#plt.figure(1) # 创建图表1
#ax1 = plt.subplot(111) # 在图表2中创建子图1
#plt.figure(2) # 创建图表2
#ax2 = plt.subplot(111) # 在图表2中创建子图1
dic={}
Numkey=0
flag=False
if os.path.exists("G:/worksp/pythonScrapy/src/process/sst/FeatureConbineOrNotprecison.txt"):
    file = codecs.open('G:/worksp/pythonScrapy/src/process/sst/FeatureConbineOrNotprecison.txt','r','utf-8')
    for line in file.readlines():
        if line.encode('utf-8').strip():
            line=line.encode('utf-8')
            if 'similar:'  in line:
                flag=True
                Numkey=re.findall("\d*\.\d*",line)[0]
                if not dic.has_key(Numkey):
                    dic[Numkey]={}
                continue
            if flag:
                if 'similar:'  in line:
                    Numkey=re.findall("\d*\.\d*",line)[0]
                    if not dic.has_key(Numkey):
                        dic[Numkey]={}
                
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
                matchObj = re.match( r'BoWFeatureExctRandomForest精度:', line, re.M|re.I)
                if matchObj:            
                    print Numkey,": BoWFeatureExctRandomForest:",line.split(':')[1]
                    if dic[Numkey].has_key('BoWFeatureExctRandomForest'):
                        dic[Numkey]['BoWFeatureExctRandomForest'].append(line.split(":")[1].strip())
                    else:
                        dic[Numkey]['BoWFeatureExctRandomForest']=[]
                        dic[Numkey]['BoWFeatureExctRandomForest'].append(line.split(":")[1].strip())
                matchObj = re.match( r'BoWFeatureExctLinearSvm精度:', line, re.M|re.I)
                if matchObj:
                    print Numkey,": BoWFeatureExctLinearSvm:",line.split(":")[1]
             
                    if dic[Numkey].has_key('BoWFeatureExctLinearSvm'):
                        dic[Numkey]['BoWFeatureExctLinearSvm'].append(line.split(":")[1].strip())
                    else:
                        dic[Numkey]['BoWFeatureExctLinearSvm']=[]
                        dic[Numkey]['BoWFeatureExctLinearSvm'].append(line.split(":")[1].strip())
          
else:
    print 'file not found,please check the filesystem'
newDic={}
y=[]
i=0
for itemKey in dic:
    newDic[itemKey]={}
    temp=[]
    for eachClassItem in dic[itemKey]:
        res=0.
        num=0
        temp.append(dic[itemKey][eachClassItem][i])
        for inum in dic[itemKey][eachClassItem]:
            res+=float(inum.strip())
            num+=1
        newDic[itemKey][eachClassItem]=res/num
    y.append(temp)
    i+=1
file.close()
print len(dic)
print dic
yFS=[]
yS=[]
yFR=[]
yR=[]

x = ['0.6','0.7', '0.8','0.9']# Make an array of x values

for itemKey in x:
    print itemKey
    for eachClassItem in dic[itemKey]:
       
        if eachClassItem=="BoWFeatureExctLinearSvm":
            yFS.append(newDic[itemKey][eachClassItem])
        if eachClassItem=="BoWLinearSvm":
            yS.append(newDic[itemKey][eachClassItem])
        if eachClassItem=='BoWFeatureExctRandomForest':
            yFR.append(newDic[itemKey][eachClassItem])
       
        if eachClassItem=="BoWRandomForest":
            yR.append(newDic[itemKey][eachClassItem])
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

plt.xlim(0.50,0.90)
plt.ylim(0.68,0.75)
plot1=plt.plot(x, yR,'ob-')# use pylab to plot x and y
#plot2=pl.plot(x, yVS,'ob--')# use pylab to plot x and y

plt.text(x[1],yR[1],u"1.BoWRandomForest",fontproperties='SimHei')
plot1=plt.plot(x, yFR,'og-')# use pylab to plot x and y
#plot2=pl.plot(x, yMF,'og--')# use pylab to plot x and y
plt.text(x[1],yFR[1],u"2.BoWFeatureExctRandomForest",fontproperties='SimHei')

plt.legend([u'1.BoWRandomForest',u'2.BoWFeatureExctRandomForest'],prop={'family':'SimHei','size':15})# make legend


plt.sca(ax2)
plt.title(u'分类的准确度Y VS 阈值X',fontproperties='SimHei')
plt.ylabel(u'Y 轴',fontproperties='SimHei')
plt.xlabel(u'X 轴',fontproperties='SimHei')

plt.xlim(0.50,0.90)
plt.ylim(0.770,0.85)
#plot1=pl.plot(x, yVF,'ob-')# use pylab to plot x and y
plot2=plt.plot(x, yS,'ob-')# use pylab to plot x and y
plt.text(x[1],yS[1],u"1.BoWLinearSvm",fontproperties='SimHei')
#plot1=pl.plot(x, yMF,'og-')# use pylab to plot x and y
plot2=plt.plot(x, yFS,'og--')# use pylab to plot x and y


plt.text(x[1],yFS[1],u"2.BoWFeatureExctLinearSvm",fontproperties='SimHei')
plt.legend([ 'BoWLinearSvm precision','BoWFeatureExctLinearSvm precision'])#,prop={'family':'SimHei','size':15})# make legend

"""    

xDim=[5000,6000,7000,8000,9000,10000,11000,12000]
plt.figure(3)
i=0
#plt.legend([ 'BoW','BoWF'],loc='upper right')
for item in x:
    y1=[]
    y2=[]
    for num in dic[item]['BoWRandomForest']:
        y1.append(float(num))
    for num in dic[item]['BoWFeatureExctRandomForest']:
        y2.append(float(num))
    ax=plt.subplot(221+i)
	
    
    plt.title(u'分类的准确度Y VS 阈值X',fontproperties='SimHei')
    plt.ylabel(u'Y 轴',fontproperties='SimHei')
    plt.xlabel(u'X 轴',fontproperties='SimHei')

    plt.xlim(5000,12000)
    plt.ylim(0.65,0.75)
    
	#plot1=pl.plot(x, yVF,'ob-')# use pylab to plot x and y
    plot1=plt.plot(xDim, y1,'ok-',label='BoWRandomForest')# use pylab to plot x and y
    plt.sca(ax)
    plt.text(xDim[1],y1[1],u"1.BoWRandomForest",fontproperties='SimHei')
	#plot1=pl.plot(x, yMF,'og-')# use pylab to plot x and y
    plot1=plt.plot(xDim, y2,'ok--',label='BoWFeatureExctRandomForest')# use pylab to plot x and y
	
	
    plt.text(xDim[4],y2[4],u"2.BoWFeatureExctRandomForest",fontproperties='SimHei')
	##,prop={'family':'SimHei','size':15})# make legend
    i=i+1
pl.subplots_adjust(top=0.94,bottom=0.10,left=0.10,right=0.95,wspace=0.35, hspace=0.4)

plt.figure(4)
i=0
#plt.legend([ 'BoW','BoWF'],loc='upper right')
for item in x:
    y1=[]
    y2=[]
    for num in dic[item]['BoWLinearSvm']:
        y1.append(float(num))
    for num in dic[item]['BoWFeatureExctLinearSvm']:
        y2.append(float(num))
    ax=plt.subplot(221+i)
	
    
    plt.title(u'分类的准确度Y VS 阈值X',fontproperties='SimHei')
    plt.ylabel(u'Y 轴',fontproperties='SimHei')
    plt.xlabel(u'X 轴',fontproperties='SimHei')

    plt.xlim(5000,12000)
    plt.ylim(0.75,0.85)
    print y1
    print xDim
	#plot1=pl.plot(x, yVF,'ob-')# use pylab to plot x and y
    plot1=plt.plot(xDim, y1,'ok-',label='BoWLinearSvm')# use pylab to plot x and y
    plt.sca(ax)
    plt.text(xDim[1],y1[1],u"1.BoWLinearSvm",fontproperties='SimHei')
	#plot1=pl.plot(x, yMF,'og-')# use pylab to plot x and y
    plot1=plt.plot(xDim, y2,'ok--',label='BoWFeatureExctLinearSvm')# use pylab to plot x and y
	
	
    plt.text(xDim[4],y2[4],u"2.BoWFeatureExctLinearSvm",fontproperties='SimHei')
	##,prop={'family':'SimHei','size':15})# make legend
    i=i+1
pl.subplots_adjust(top=0.94,bottom=0.10,left=0.10,right=0.95,wspace=0.35, hspace=0.4)



pl.show()# show t

