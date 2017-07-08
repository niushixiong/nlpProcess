# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 16:07:39 2016

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
else:
    print 'file not found,please check the filesystem'
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
#yVsum=[]
#yMsum=[]
#ysum=[]
#for i in range(len(yS)):
#    yVsum.append((yVF[i]+yVS[i])/2)
#    yMsum.append((yMF[i]+yMS[i])/2)
#    ysum.append((yF[i]+yS[i])/2)


plt.sca(ax1)
plt.title(u"分类的准确度Y VS 阈值X",fontproperties='SimHei')
plt.ylabel(u'Y 轴',fontproperties='SimHei')
plt.xlabel(u'X 轴',fontproperties='SimHei')

plt.xlim(0.550,0.950)
plt.ylim(0.675,0.725)
plot1,=plt.plot(x, yVF,'ok-')# use pylab to plot x and y
#plot2=pl.plot(x, yVS,'ob--')# use pylab to plot x and y

plt.text(x[4],yVF[4],u"1.BoWVRandomForest",fontproperties='SimHei')
plot2,=plt.plot(x, yMF,'ok--')# use pylab to plot x and y
#plot2=pl.plot(x, yMF,'og--')# use pylab to plot x and y
plt.text(x[4],yMF[4],u"2.BoWMRandomForest",fontproperties='SimHei')

plot3,=plt.plot(x, yF,'ok-.')# use pylab to plot x and y
#plot2=pl.plot(x, yF,'or--')# use pylab to plot x and y
plt.text(x[4],yF[4],u"3.BoWRandomForest",fontproperties='SimHei')
# 为第一个线条创建图例
#first_legend = plt.legend(handles=[plot1])

# 手动将图例添加到当前轴域
#ax = plt.gca().add_artist(first_legend)

# 为第二个线条创建另一个图例
#plt.legend(handles=[plot2])


# 为第二个线条创建另一个图例
#plt.legend(handles=[plot3])


#plt.legend(handles=[plot1,plot2,plot3])#,label=[u'1.BoWVRandomForest',u'2.BoWMRandomForest',u'3.BoWRandomForest'],prop={'family':'SimHei','size':15})# make legend
plt.legend(handles=[plot1,plot2,plot3],labels=[u'1.BoWVRandomForest',u'2.BoWMRandomForest',u'3.BoWRandomForest'])

plt.sca(ax2)
plt.title(u'分类的准确度Y VS 阈值X',fontproperties='SimHei')
plt.ylabel(u'Y 轴',fontproperties='SimHei')
plt.xlabel(u'X 轴',fontproperties='SimHei')

plt.xlim(0.550,0.950)
plt.ylim(0.770,0.795)
#plot1=pl.plot(x, yVF,'ob-')# use pylab to plot x and y
plot4,=plt.plot(x, yVS,'ok-')# use pylab to plot x and y
plt.text(x[4],yVS[4],u"1.BoWVLinearSvm",fontproperties='SimHei')
#plot1=pl.plot(x, yMF,'og-')# use pylab to plot x and y
plot5,=plt.plot(x, yMS,'ok--')# use pylab to plot x and y
plt.text(x[4],yMS[4],u"2.BoWMLinearSvm",fontproperties='SimHei')

#plot1=pl.plot(x, yF,'or-')# use pylab to plot x and y
plot6,=plt.plot(x, yS,'ok:')# use pylab to plot x and y
plt.text(x[7],yS[7],u"3.BoWLinearSvm",fontproperties='SimHei')

plt.legend(handles=[plot4,plot5,plot6],labels=[u'BoWVLinearSvm precision', 'BoWMLinearSvm precision','BoWLinearSvm precision'])#,prop={'family':'SimHei','size':15})# make legend

pl.show()# show t

