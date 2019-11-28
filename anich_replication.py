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

The tests were tailored to each grade
level. Because there are no natural units for the test results, I
scaled the test scores into percentile ranks. Specifically, in each
grade level the regular and regular/aide students were pooled
together, and students were assigned percentile scores based on
their raw test scores, ranging from 0 (lowest score) to 100 (highest
score). A separate percentile distribution was generated for each
subject test (e.g., Math-SAT, Reading-SAT, Word-SAT, etc.). For
each test I then determined where in the distribution of the
regular-class students every student in the small classes would
fall, and the students in the small classes were assigned these
percentile scores. Finally, to summarize overall achievement, the
average of the three SAT percentile rankings was calculated. If
the performance of students in the small classes was distributed
in the same way as performance of students in the regular classes,
the average percentile score for students in the small classes
would be 50.

Formally, denote the cumulative distribution of scores on test j (denoted
Tj) of students in the regular and regular/aide classes as FR(Tj) 5 prob [TRi
j ,
Tj] 5 yj. For each student i in a small class, we then calculated FR(TSi
j ) 5 ySi
j .
Naturally, the distribution of yj for students in regular classes follows a uniform
distribution. We then calculated the average of the three (or two for BSF)
percentile rankings for each student. If one subtest score was missing, we took the
average of the two percentiles that were available; and if two were missing, we
used the percentile score corresponding to the only available test.
"""
#calculate percentiles for each test
for i in ['k', '1', '2', '3']:
    tests = [
            f'g{i}treadss',
            f'g{i}tmathss',
    ]
    test_pc = []
    #wordskills test not present in third grade
    if i != '3':
        tests.append(f'g{i}wordskillss')
    for test in tests:
        raw_scores = data[[test, f'g{i}classtype']].loc[data[test].notna() == True]
        percentiles = pd.DataFrame(
                index = raw_scores.index,
                columns = [f'{test}_pc',],
        )
        
        #calculate reg and reg + aide distribution
        reg_scores = raw_scores.loc[raw_scores[f'g{i}classtype'] != 'SMALL CLASS']
        n = reg_scores[test].count()
        for j, row in reg_scores.iterrows():
            below = reg_scores.loc[reg_scores[test] < row[test]]
            below = below[test].count()
            percentile = below/n
            percentiles[f'{test}_pc'][j] = percentile
        
        #fit small scores into distribution
        small_scores = raw_scores.loc[raw_scores[f'g{i}classtype'] == 'SMALL CLASS']
        for j, row in small_scores.iterrows():
            below = reg_scores.loc[reg_scores[test] < row[test]]
            below = below[test].count()
            percentile = below/n
            percentiles[f'{test}_pc'][j] = percentile
        
        data = pd.concat([data, percentiles], axis = 1, sort = True)
        test_pc.append(f'{test}_pc')
    
    #calculate average percentiles for each year
    all_scores = data[test_pc]
    avg_percentile = pd.DataFrame(
            index = all_scores.index,
            columns = [f'g{i}_avg_pc',],
    )
    for k, row in all_scores.iterrows():
        if np.all(row.isna()):
            continue
        else:
            avg = row.mean()
            avg_percentile[f'g{i}_avg_pc'][k] = avg
    data = pd.concat([data, avg_percentile], axis = 1, sort = True)
        
#calculate attrition dummy (left = 1, never left = 0)
reference_df = data[['gkclasstype',
                    'g1classtype',
                    'g2classtype',
                    'g3classtype',
]]
attrition = pd.DataFrame(
        index = reference_df.index,
        columns = ['attrition',],
)
for i, row in reference_df.iterrows():
    if np.all(row.notna()):
        attrition['attrition'][i] = 0
    else:
        test = row.isna()
        enter = False
        leave = False
        for j, item in row.notna().iteritems():
            if enter == False:
                if item == True:
                    enter = True
            elif leave == False:
                if item == False:
                    leave = True
            else:
                break
        if leave == True:
            attrition['attrition'][i] = 1
        else:
            attrition['attrition'][i] = 0
data = pd.concat([data, attrition], axis = 1, sort = True)

            
        
        

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

table_i_data = {}
columns = []
for i in ['k', '1', '2', '3']:
    
    #split up data by study entrance year
    class_type = f'g{i}classtype'
    columns.append(class_type)
    keep = data.loc[data[columns[len(columns)-1]].notna() == True]
    for column in columns[:-1]:
        keep = keep.loc[keep[column].isna() == True]
    table_i_data[f'enter_{i}'] = keep
    
    
    

     
    