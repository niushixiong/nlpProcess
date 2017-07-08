#-*-coding:utf-8
'''
Created on 2016年6月12日

@author: sx
'''
import numpy as np
from numpy import vstack
text=['A61C1/08','A61B5/0245','A61B5/11','H04M1/12','H04N7/18','A63F13/24','A63F13/90','H04M1/02']
text.append("asdsad")
print text
A = np.array([[1, 2, 3],[2,2,3]])
B = np.array([4, 4, 5])

c=vstack((A,B))
print c
if text.__contains__('A61B5/024'):
    print "contains"
else:
    print "not contain"