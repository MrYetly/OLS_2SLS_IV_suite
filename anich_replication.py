#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:44:45 2019

@author: ianich
"""

from anich_tools import Regression
import pandas as pd
import numpy as np
import os

#set current directory
__location__ = os.getcwd()

#import data
filenames = os.listdir(__location__+'/data')
raw = {}
for filename in filenames:
    df = pd.read_stata(__location__+'/data/'+filename)
    df_name = filename[:-4]
    raw[df_name] = df
data = raw['STAR_Students']

"""
Percentiles

"""
#calculate percentiles

"""
Table I

summary stats for students who entered in each year

Variables Needed:
    - Free lunch
        - STAR_Students
            - 'gkfreelunch'
    - White/Asian
        - STAR_Students    
            - 'race'
    - Age in 1985
        - STAR_Student
            - 'birthmonth'
            - 'birthday'
            - 'birthyear'
    - Attrition Rate
        - STAR_Students
            - if they left the program at any time
    - class size in year
        - STAR_Students
            - 'gkclasssize'
    - Percentile score in year
        - STAR_Students: math reading word
            - 'gktmathss'
            - 'gktreadss'
            - 'gktwordskillss'
        

For Groups by year:
    - small class
    - reg
    - reg + aide
        = STAR_Students
            - 'gkclasstype'
    
P-value:
    - for anova F-test
"""
#recreate Table I
table_data = {}
columns = []
for i in ['k', '1', '2', '3']:
    class_type = f'g{i}classtype'
    columns.append(class_type)
    keep = data.loc[data[columns[len(columns)-1]].notna() == True]
    for column in columns[:-1]:
        keep = keep.loc[keep[column].isna() == True]
    keep = keep[]
    table_data[f'enter_{i}'] = keep
    

     
    