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
Build variables/dummies
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

#calculate age in '85
age_85 = 1985 - data[['birthyear']] 
data.insert(len(data.columns), 'age_85', age_85)

#calculate White/Asian dummy
wa = data['race'].copy()
wa.replace(['white', 'asian'], 1, inplace = True)
wa.replace(
        [
                'black',
                'hispanic',
                'NATIVE AMERICAN', 
                'other',
        ],
        0,
        inplace = True,
)
data.insert(len(data.columns), 'white/asian', wa)

#calculate free lunch dummy
for i in ['k', '1', '2', '3']:
    dummy = data[f'g{i}freelunch'].copy()
    dummy.replace('FREE LUNCH', 1, inplace = True)
    dummy.replace('NON-FREE LUNCH', 0, inplace = True)
    data.insert(len(data.columns), f'{i}_freelunch', dummy)
"""

- create class type dummies (small) (reg+Aide)
"""

#create dummies to control for school effects
school_effects = {}
for i in ['k', '1', '2', '3']:
    schids = list(data[f'g{i}schid'].loc[data[f'g{i}schid'].notna() == True].unique())
    #drop one id to prevent perfect multicolinearity
    schids.pop()
    year_list = []
    for schid in schids:
        dummy = data[f'g{i}schid'].copy()
        dummy.replace(schid, 1, inplace = True)
        dummy = dummy.loc[dummy.notna() == True]
        dummy.where(dummy == 1, other = 0, inplace = True)
        name = f'd_g{i}{schid}'
        data.insert(len(data.columns), name, dummy)
        year_list.append(name)
    school_effects[f'g{i}'] = year_list

        
#create class-type dummies


"""
Table I
"""

#recreate Table I
table_i_data = {}
columns = []
label_grade = {
            'k': 'kindergarten',
            '1': 'first grade',
            '2': 'second grade',
            '3': 'third grade',
    }
label_type = {
        'Regular/Aide': 'REGULAR + AIDE CLASS',
        'Small': 'SMALL CLASS',
        'Regular': 'REGULAR CLASS',
        }
sub_tables = {}
for i in ['k', '1', '2', '3']:
    
    #split up data by study entrance year
    class_type = f'g{i}classtype'
    columns.append(class_type)
    keep = data.loc[data[columns[len(columns)-1]].notna() == True]
    for column in columns[:-1]:
        keep = keep.loc[keep[column].isna() == True]
    table_i_data[f'enter_{i}'] = keep
    
    sub_index = [1,2,3,4,5,6]
    super_label = f'Students who entered STAR in {label_grade[i]}'
    super_index = [super_label for i in sub_index]
    multi_index = pd.MultiIndex.from_arrays([super_index, sub_index])
    
    #create sub-table
    label_vars = {
            'Free lunch': f'{i}_freelunch',
            'White/Asian': 'white/asian',
            'Age in 1985': 'age_85',
            'Attrition rate': 'attrition',
            f'Class size in {label_grade[i]}': f'g{i}classsize',
            f'Percentile score in {label_grade[i]}': f'g{i}_avg_pc',
        }
    sub_table = pd.DataFrame(
            columns = [
                    'Variable',
                    'Small',
                    'Regular',
                    'Regular/Aide',
                    'Joint P-value',
            ],
            index = multi_index
    )
    sub_table['Variable'] = list(label_vars.keys())
    
    #insert means into sub-table
    group_df = table_i_data[f'enter_{i}']
    for label_c, class_t in label_type.items():  
        type_df = group_df.loc[group_df[f'g{i}classtype'] == class_t]
        for label_v, var in label_vars.items():
            entry = type_df[var].mean()
            sub_table.loc[sub_table['Variable'] == label_v, label_c] = entry
    #insert f-test p-values
    p_func = Regression(group_df)
    for label_v, var in label_vars.items():
        F, p = p_func.f_test(
                'anova',
                var_to_analyze = var,
                grouping_var = class_type,
        )
        entry = p
        sub_table.loc[sub_table['Variable'] == label_v, 'Joint P-value'] = entry
    sub_tables[f'enter_{i}'] = sub_table

#concatenate all sub-tables intro table I
table_i = pd.DataFrame()
for name, sub_table in sub_tables.items():
    table_i = pd.concat([table_i, sub_table])
#output table I
table_i.to_csv('tables_figues/table_I.csv')
     
"""
Table II
"""

"""
- regress six table 1 variables against dummies for class size, school effects
- table p-values for the F test that assignement to a class type has no effect
on the explanatory variables.

"""


