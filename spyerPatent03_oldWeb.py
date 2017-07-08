# -*- coding: utf-8 -*-
"""
Created on Fri Mar 04 20:29:08 2016

@author: shixiong
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import urllib
import urllib2
import cookielib
import webbrowser
import codecs
import time,os
import re
from selenium import webdriver 
class patentSpyer:
    def __init__(self):
        self.url="http://dbpub.cnki.net/grid2008/dbpub/brief.aspx?id=scpd"
        #self.proxyURL=""
# post data header        
        self.Header={
        'Host':'dbpub.cnki.net',
        'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Referer':'http://dbpub.cnki.net/grid2008/dbpub/brief.aspx?id=scpd',
        'Content-Type':'application/x-www-form-urlencoded',
        'Connection':'Keep-Alive'
    
            }
        self.post={  
        'advancedfield1':'专利名称',
        'advancedvalue1':'电路板',
        'bCurYearTempDB':'1',
        'display'	:'chinese',
        'encode':'gb',
        'fieldtips':'篇名/[在文献标题中检索。对该检索项的检索是按词进行的，请尽可能输入完整的词，以避免漏检。],关键词/[检索文献的关键词中满足检索条件的文献。对该检索项的检索是按词进行的，请尽可能输入完整的词，以避免漏检。],第一责任人/[请选择检索项并指定相应的检索词，选择排序方式、匹配模式、文献时间等限定条件，然后点击“检索”。],作者/[可输入作者完整姓名，或只输入连续的一部分。],机构/[可输入完整的机构名称，或只输入连续的一部分。],中文摘要/[对该检索项的检索是按词进行的，请尽可能输入完整的词，以避免漏检。],引文/[请选择检索项并指定相应的检索词，选择排序方式、匹配模式、文献时间等限定条件，然后点击“检索”。],全文/请选择检索项并指定相应的检索词，选择排序方式、匹配模式、文献时间等限定条件，然后点击“检索”。],基金/[检索受满足条件的基金资助的文献。],中文刊名/[请输入部分或全部刊名。],ISSN/[请输入完整的ISSN号。],年/[输入四位数字的年份。],期/[输入期刊的期号，如果不足两位数字，请在前面补“0”，如“08”。],主题/[主题包括篇名、关键词、中文摘要。可检索出这三项中任一项或多项满足指定检索条件的文献。对主题是按词检索的，请尽可能输入完整的词，以避免漏检。]',
        'hdnFathorCode':'sysAll',
        'hdnIsAll':'true',
        'hdnSearchType':'',	
        'hdnUSPSubDB':'专利类别,+1+2+3+,3,3',
        'ID':'scpd',
        'imageField.x':'0',
        'imageField.y':'0',
        'MM_fieldName':'申请日@@@申请日@@@公开日@@@公开日@@@更新日期@@@更新日期',
        'MM_fieldValue_1_1':'',
        'MM_fieldValue_1_2':	'',
        'MM_fieldValue_2_1':	'',	
        'MM_fieldValue_2_2':	'',	
        'MM_hiddenRelation':'>=@@@<=@@@>=@@@<=@@@>=@@@<=',
        'MM_hiddenTxtName':'MM_fieldValue_1_1@@@MM_fieldValue_1_2@@@MM_fieldValue_2_1@@@MM_fieldValue_2_2@@@MM_Update_Time@@@MM_Update_EndTime',
        'MM_slt_updateTime':	'',
        'MM_Update_EndTime':	'',
        'MM_Update_Time':'',
        'NaviDatabaseName':'SCPD_ZJCLS',
        'NaviField':'专题子栏目代码',
        'order':'dec',
        'RecordsPerPage':'20',
        'searchAttachCondition':'',
        'SearchFieldRelationDirectory':'',
        'searchmatch':'0',
        'SearchQueryID':'',	
        'selectbox':'A',
        'selectbox':'B',
        'selectbox':'C',
        'selectbox':'D',
        'selectbox':'I',
        'selectbox':'H',
        'selectbox':'F',
        'selectbox':'E',
        'singleleafcode':'',	
        'strNavigatorName':',基础科学,工程科技Ⅰ辑,工程科技Ⅱ辑,农业科技,医药卫生科技,哲学与人文科学,社会科学Ⅱ辑,信息科技',
        'strNavigatorValue':',A,B,C,D,E,F,H,I',
        'systemno	':'',
        'TablePrefix':'SCPD',
        'TableType':'PY',
        'updateTempDB':'',	
        'userright':'',	
        'VarNum':	'1',
        'View':'SCPD',
        'yearFieldName':'年'
        }
        self.cookie=0
        self.opener=0
        
    #def getCookieOpener():
        
    def readHtml(self):

        self.cookie = cookielib.CookieJar()  
        self.opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(self.cookie))
        #需要POST的数据#
        postdata=urllib.urlencode(self.post)
        #自定义一个请求#
        req = urllib2.Request(  
            url = self.url,  
            data = postdata
        )
        #访问该链接#
        result = self.opener.open(req)
        #打印返回的内容#
        #print result.read()
        return result
        
    def parserElement(self):
        
        self.cookie = cookielib.CookieJar()  
        self.opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(self.cookie))
        #需要POST的数据#
        postdata=urllib.urlencode(self.post)
        #自定义一个请求#
        req = urllib2.Request(  
            url = self.url,  
            data = postdata
        )
        #访问该链接#
        result = self.opener.open(req)
        print self.cookie
        htmlSource=result.read()
        #testurl=driver.current_url()
       
        file_object = codecs.open('data2.html', 'w','utf-8')
  #  print all_the_text
        txt=unicode(htmlSource,"utf-8")
   # file_object.write(u'中文')
        file_object.write(txt)
        print self.cookie
        driver=webdriver.Chrome() 
        
        driver.get(self.url)
        print self.cookie
       
        driver.get(self.url)
       # search=driver.find_element_by_name('advancedvalue1')
        #button = driver.find_element(By.xpath(//divid_grid_save2')
    
        quanxuan=driver.find_elements_by_xpath("//a[contains(@href,'checkAll')]")
        driver.delete_all_cookies()
        quanxuan[1].click()
        
       
        
    
        for ck  in self.cookie:
            print type({ck.name:ck.value})
            driver.add_cookie({ck.name:ck.value})
        
      
       # driver.execute_script('$(arguments[0]).fadeOut()',bt)
        #time.sleep(5)
        save=driver.find_elements_by_xpath("//a[contains(@href,'windowOpener')]")
        partialurl=save[1].get_attribute("href")
        print partialurl
        pattern=re.compile(r'\(')
        split_list=pattern.split(partialurl)
        print split_list[1]
        fullurl='http://dbpub.cnki.net'+split_list[1][1:-2]
        print fullurl
        driver.get(fullurl)
       # list_content=opener.open(fullurl)
       # content=list_content.read()
        
        time.sleep(10)
        driver.close()
            
            
if __name__=='__main__':
    print "the spyer start:"
    pa=patentSpyer()
    result=pa.parserElement()  

    print "the spyder end"
    
# 浏览器读取html find element 触发js函数
   
