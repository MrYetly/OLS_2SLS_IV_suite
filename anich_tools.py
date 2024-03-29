#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:24:59 2019

@author: ianich
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from scipy.stats import f
"""
#testing setup
#Create test data w/ known coefficients
X, y, coef = make_regression(n_samples = 1000, n_features = 10, coef = True)
test_data = pd.DataFrame(X)
control_vars = []
for i in range(len(test_data.columns)):
    control_vars.append(f'control_{i}')
test_data.columns = pd.Series(control_vars)
test_data.insert(0, 'outcome', y)
test_data_coef = pd.DataFrame(
        coef,
        columns = ['coefficient',],
        index = control_vars,
)
"""

class Regression():
    def __init__(self, dataframe):
        #general attribtues
        self.dataframe = dataframe.copy()
        self.intercept = None
        self.control_names = None
        self.outcomes = None
        self.controls = None
        self.nk_check = None
        self.pmc_check = None
        #OLS attributes
        self.y = None
        self.X = None
        self.n = None
        self.k = None
        self.res = None
        self.coef = None
        self.y_fitted = None
        self.se = None
        self.HC1 = None
        self.R2 = None
        #cluster OLS attributes
        self.G = None
        self.groups = None
        self.cluster_se = None
        self.cluster_HC1 = None
        #2SLS attributes
        self.controls_fitted = None
        self.X_hat = None
        self.endog_name = None
        self.endog = None
        self.endog_fitted = None
        self.exog_name = None
        self.exog = None
        self.instrument_name = None
        self.instrument = None
        self.Z = None
        self.fs_controls = None
        self.fs_coef = None
        self.fs_f_stat = None
    
    def ols(self, outcomes, controls, intercept = True):
        #need to add auto drop for pmc failure
        self.nk_check = None
        self.pmc_check = None
        self.control_names = controls[:]
        if intercept:
            self.intercept = True
            self.control_names.append('intercept')
            if 'intercept' not in list(self.dataframe.columns):
                self.dataframe.insert(
                        len(self.dataframe.columns),
                        'intercept',
                        1
                )
        #only include observations that have an outcome
        outcomes = self.dataframe[outcomes]
        outcomes = outcomes.loc[outcomes.notna() == True]
        self.outcomes = outcomes
        self.controls = self.dataframe.loc[list(self.outcomes.index), self.control_names]
        self.y = np.matrix(self.outcomes.values).T
        self.X = np.matrix(self.controls.values)
        self.n = self.X.shape[0]
        self.k = self.X.shape[1]
        if self.n < self.k:
            self.nk_check = False
        else:
            self.nk_check = True
        try:
            (self.X.T*self.X).I
            self.pmc_check = True
        except Exception as e:
            self.pmc_check = e
        #test invertability of (X'X)^-1
        if self.pmc_check != True:
            print(f'error: {self.pmc_check}')
        elif self.nk_check == False:
            print('error: number of observations < number of controls')            
        else:
            X = self.X
            y = self.y
            n = self.n
            k = self.k
            
            #use matrix methods to run OLS
            B = (X.T*X).I*X.T*y
            self.coef = pd.DataFrame(
                    B,
                    columns = ['coefficient',],
                    index = self.controls.columns,
            )
            
            P = X*(X.T*X).I*X.T
            y_hat = P*y
            self.y_fitted = y_hat
            
            M = (np.identity(n)-P)
            e_hat = M*y
            self.res = e_hat
            
            #calculate HC1 covariance matrix estimator/standard errors
            XtDX = 0
            for i in range(n):
                x = X[i]
                e = e_hat[i,0]
                e2 = e*e
                product = x.T*x*e2
                XtDX += product
            self.HC1 = (n/(n-k))*(X.T*X).I*XtDX*(X.T*X).I
            se = [np.sqrt(self.HC1[i,i]) for i in range(k)]
            self.se = pd.DataFrame(
                    se, 
                    columns = ['Standard Errors (HC1)'],
                    index = self.controls.columns,
            )
            #calculate R-squared
            sse = 0
            for i in range(n):
                square = self.res[i,0]**2
                sse += square
            mdsquare = 0
            y_mean = self.outcomes.mean()
            for i in range(n):
                square = (y[i, 0] - y_mean)**2
                mdsquare += square
            R2 = 1 - (sse/mdsquare)
            self.R2 = R2
                        
    def cluster_ols(self, outcomes, controls, grouping_var = None, intercept = True):
        self.ols(outcomes, controls, intercept = True)
        if self.pmc_check == True and self.nk_check == True:
            #calculate cluster robust HC1/standard errors
            self.groups = {}
            group_list = list(
                    self.dataframe[grouping_var].loc[
                            self.dataframe[grouping_var].notna() == True
                    ].unique()
            )
            for group in group_list:
                all_vars = list(self.controls.columns)
                all_vars.append(self.outcomes.name)
                group_df = self.dataframe.loc[
                        self.dataframe[grouping_var] == group,
                        all_vars,
                ].copy()
                self.groups[f'{group}'] = group_df
            n = self.n
            k = self.k
            G = len(self.groups)
            self.G = G
            Omega_hat = np.zeros((k,k))
            #for loop for Omega_hat summation
            for group, group_df in self.groups.items():
                #only keep obs that have outcomes
                out = group_df[self.outcomes.name]
                out = out.loc[out.notna() == True]
                y = np.matrix(out.values).T
                dep = group_df[list(self.controls.columns)]
                dep = dep.loc[list(out.index)]
                X = np.matrix(dep.values)
                print(X)
                try:
                    M = (np.identity(X.shape[0]) - X*(X.T*X).I*X.T)
                    e_hat = M*y
                    product = X.T*e_hat*e_hat.T*X
                    Omega_hat = Omega_hat + product
                except Exception as e:
                    print(f'group: {group}\n', e)
            X = self.X
            a = ((n-1)/(n-k))*(G/(G-1))
            cluster_HC1 = a*(X.T*X).I*Omega_hat*(X.T*X).I
            self.cluster_HC1 = cluster_HC1
            cluster_se = [np.sqrt(self.cluster_HC1[i,i]) for i in range(k)]
            self.cluster_se = pd.DataFrame(
                    cluster_se, 
                    columns = ['Cluster Robust Standard Errors (HC1)'],
                    index = self.controls.columns,
            )
    
    def f_test(self, test_type, null_controls = None, var_to_analyze = None, grouping_var = None):
        if null_controls == None and (var_to_analyze == None and grouping_var == None):
            print('error: no test parameters specified')
        elif test_type == 'null':
            if self.pmc_check != True:
                print('No stored data. Run a regression method first.')
            else:
                n = self.n
                k = self.k
                q = len(null_controls)
                #sum unrestricted residuals
                sse_ur = 0
                for i in range(n):
                    square = self.res[i,0]**2
                    sse_ur += square
                #create and sum restricted residuals
                restricted_controls = set(self.control_names)
                null_controls = set(null_controls)
                restricted_controls = restricted_controls - null_controls
                restricted_controls = list(restricted_controls)
                X_r = np.matrix(self.controls[restricted_controls].values)
                M_r = np.identity(n) - X_r*(X_r.T*X_r).I*X_r.T
                y = self.y
                res_r = M_r*y
                sse_r = 0
                for i in range(n):
                    square = res_r[i,0]**2
                    sse_r += square
                #calculate F-stat, p-value
                F = ((sse_r - sse_ur)/sse_ur)*((n-k)/q)
                p = 1 - f.cdf(F, q, n-k)
                return F, p
        elif test_type == 'anova':
            dataframe = self.dataframe[[var_to_analyze, grouping_var]].copy()
            N = dataframe[var_to_analyze].count()
            #define groups
            groups = list(
                    dataframe[grouping_var].loc[
                            dataframe[grouping_var].notna() == True
                    ].unique()
            )
            K = len(groups)
            sample_mean = dataframe[var_to_analyze].mean()
            #calculate relevant sums
            gm_sm_sq_sum = 0
            in_group_sq_sum = 0
            for group in groups:
                group_df = dataframe.loc[dataframe[grouping_var] == group]
                group_mean = group_df[var_to_analyze].mean()
                n_g = group_df[var_to_analyze].count()
                #for loop to sum in-group mean difference squared
                in_group_sq = 0
                column = group_df.loc[
                        group_df[var_to_analyze].notna() == True,
                        var_to_analyze
                ]
                for i, item in column.iteritems():
                    y = item
                    mean_dif_sq = (y - group_mean)**2
                    in_group_sq += mean_dif_sq
                in_group_sq_sum += in_group_sq
                #summimg size of group * (group mean - sample_mean)^2
                gm_sm_sq = n_g*((group_mean - sample_mean)**2)
                gm_sm_sq_sum += gm_sm_sq
            #calculate F-stat, p-value
            F = ((N-K)/(K-1))*(gm_sm_sq_sum/in_group_sq_sum)
            p = 1 - f.cdf(F, K-1, N-K)
            return F, p
            
    def iv(self, 
           outcomes, 
           controls, 
           endog = None, 
           instruments = None, 
           intercept = True
           ):
        #sort info into attributes
        self.control_names = controls[:]
        self.nk_check = None
        self.pmc_check = None
        if intercept:
            self.intercept = True
            self.control_names.append('intercept')
            if 'intercept' not in list(self.dataframe.columns):
                self.dataframe.insert(
                        len(self.dataframe.columns),
                        'intercept',
                        1
                )
        #only include observations that have an outcome
        outcomes = self.dataframe[outcomes]
        outcomes = outcomes.loc[outcomes.notna() == True]
        self.outcomes = outcomes
        self.y = np.matrix(self.outcomes.values).T
        self.controls = self.dataframe.loc[list(self.outcomes.index), self.control_names]
        self.X = np.matrix(self.controls.values)
        self.endog_name = endog
        self.endog = self.controls[self.endog_name]
        self.instrument_names = instruments[:]
        instruments = self.dataframe[self.instrument_names]
        instruments = instruments.loc[list(self.controls.index)]
        self.instruments = instruments
        exog = self.control_names[:]
        exog.remove(self.endog_name)
        self.exog_name = exog
        self.exog = self.controls[self.exog_name]
        fs_controls = self.exog.copy()
        fs_controls = pd.concat(
                [fs_controls, self.instruments],
                axis = 1,
                sort = True,
        )
        self.fs_controls = fs_controls
        self.Z = np.matrix(self.fs_controls.values)
        n_fs = self.Z.shape[0]
        k_fs = self.Z.shape[1]
        if n_fs < k_fs:
            self.nk_check = False
        else:
            self.nk_check = True
        try:
            (self.Z.T*self.Z).I
            self.pmc_check = True
        except Exception as e:
            self.pmc_check = e
        #test invertability of (Z'Z)^-1
        if self.pmc_check != True:
            print(f'error first stage: {self.pmc_check}')
        elif self.nk_check == False:
            print('error first stage: number of observations < number of controls')
        else:
            #begin first stage regression
            Z = self.Z 
            X_endog = np.matrix(self.endog.values).T
            print(np.any(np.isnan(Z)))
            print(np.any(np.isnan(X_endog)))
            #first stage coefficient
            B_fs = (Z.T*Z).I*Z.T*X_endog
            self.fs_coef = pd.DataFrame(
                    B_fs,
                    columns = ['first stage coefficient',],
                    index = self.fs_controls.columns,
            )

            #endogenous fitted values and first stage residuals
            P_z = Z*(Z.T*Z).I*Z.T
            X_endog_hat = P_z*X_endog
            fs_res = (np.identity(n_fs) - P_z)*X_endog
            self.endog_fitted = pd.Series(
                    list(X_endog_hat.flat),
                    name = f'{self.endog_name}_fitted',
                    index = list(self.endog.index),
            )
    
            #calculate first stage f-stat
            q = len(self.instrument_names)
            #sum unrestricted residuals
            sse_ur = 0
            for i in range(n_fs):
                square = fs_res[i,0]**2
                sse_ur += square
            #create and sum restricted residuals
            Z_r = np.matrix(self.exog.values)
            M_r = np.identity(n_fs) - Z_r*(Z_r.T*Z_r).I*Z_r.T
            res_r = M_r*X_endog
            sse_r = 0
            for i in range(n_fs):
                square = res_r[i,0]**2
                sse_r += square
            #calculate F-stat, p-value
            F = ((sse_r - sse_ur)/sse_ur)*((n_fs-k_fs)/q)
            self.fs_f_stat = F
            
        
            #begin second stage regression
            controls_fitted = self.exog.copy()
            controls_fitted.insert(
                    0,
                    self.endog_fitted.name,
                    self.endog_fitted,
            )
            self.controls_fitted = controls_fitted
            self.X_hat = np.matrix(self.controls_fitted.values)
            self.n = self.X_hat.shape[0]
            self.k = self.X_hat.shape[1]
            if self.n < self.k:
                self.nk_check = False
            else:
                self.nk_check = True
            try:
                (self.X_hat.T*self.X_hat).I
                self.pmc_check = True
            except Exception as e:
                self.pmc_check = e
            #test invertability of (X'X)^-1
            if self.pmc_check != True:
                print(f'error second stage: {self.pmc_check}')
            elif self.nk_check == False:
                print('error second stage: number of observations < number of controls')
            else:
                #note: X here is the exogenous vars plus fitted endogenous var
                X_hat = self.X_hat
                y = self.y
                n = self.n
                k = self.k
                
                B = (X_hat.T*X_hat).I*X_hat.T*y
                self.coef = pd.DataFrame(
                        B,
                        columns = ['2SLS coefficient',],
                        index = self.controls_fitted.columns,
                )
                
                P = X_hat*(X_hat.T*X_hat).I*X_hat.T
                y_hat = P*y
                self.y_fitted = y_hat
                
                M = np.identity(n)-P
                e_hat = M*y
                self.res = e_hat
                
                #calculate 2SLS HC1 covariance matrix estimator/standard errors
                XtDX = 0
                for i in range(n):
                    x = X_hat[i]
                    e = e_hat[i,0]
                    e2 = e*e
                    product = x.T*x*e2
                    XtDX += product
                self.HC1 = (n/(n-k))*(X_hat.T*X_hat).I*XtDX*(X_hat.T*X_hat).I
                se = [np.sqrt(self.HC1[i,i]) for i in range(k)]
                self.se = pd.DataFrame(
                        se, 
                        columns = ['2SLS Standard Errors (HC1)'],
                        index = self.controls_fitted.columns,
                )
                
                #calculate R-squared
                sse = 0
                for i in range(n):
                    square = self.res[i,0]**2
                    sse += square
                mdsquare = 0
                y_mean = self.outcomes.mean()
                for i in range(n):
                    square = (y[i, 0] - y_mean)**2
                    mdsquare += square
                R2 = 1 - (sse/mdsquare)
                self.R2 = R2
                
    
    def cluster_iv(self,
                   outcomes, 
                   controls, 
                   endog = None, 
                   instrument = None, 
                   grouping_var = None, 
                   intercept = True,
                   ):
        self.iv(outcomes, 
                controls, 
                endog = endog, 
                instrument = instrument, 
                intercept = True
        )
        if self.pmc_check == True and self.nk_check == True:
            #calculate cluster robust HC1/standard errors
            self.groups = {}
            all_vars = self.controls_fitted.copy()
            all_vars.insert(0, self.outcomes.name, self.outcomes)
            grouping_series = self.dataframe[grouping_var].loc[list(all_vars.index)]
            all_vars.insert(0, grouping_var, grouping_series)
            group_list = list(
                    all_vars[grouping_var].loc[
                            all_vars[grouping_var].notna() == True
                    ].unique()
            )
            for group in group_list:
                group_df = all_vars.loc[all_vars[grouping_var] == group].copy()
                self.groups[f'{group}'] = group_df
            n = self.n
            k = self.k
            G = len(self.groups)
            self.G = G
            Omega_hat = np.zeros((k,k))
            #for loop for Omega_hat summation
            for group, group_df in self.groups.items():
                
                X = np.matrix(group_df[list(self.controls_fitted.columns)].values)
                y = np.matrix(group_df[self.outcomes.name].values).T
                try:
                    M = (np.identity(X.shape[0]) - X*(X.T*X).I*X.T)
                    e_hat = M*y
                    product = X.T*e_hat*e_hat.T*X
                    Omega_hat = Omega_hat + product
                except Exception as e:
                    print(f'group: {group}\n', e)
            X_hat = self.X_hat
            a = ((n-1)/(n-k))*(G/(G-1))
            cluster_HC1 = a*(X_hat.T*X_hat).I*Omega_hat*(X_hat.T*X_hat).I
            self.cluster_HC1 = cluster_HC1
            cluster_se = [np.sqrt(self.cluster_HC1[i,i]) for i in range(k)]
            self.cluster_se = pd.DataFrame(
                    cluster_se, 
                    columns = ['2SLS Cluster Robust Standard Errors (HC1)'],
                    index = self.controls.columns,
            )
                

#OLS testing
"""
reg = Regression(test_data)
reg.ols('outcome', control_vars, intercept = True)
print('ols\n', reg.coef, '\nR2\n', reg.R2)
print('\ndummy\n', test_data_coef) 
print(reg.se) 
F, p = reg.f_test(test_type = 'null', null_controls = ['control_0', 'control_1'])
print(f'\nF-stat: {F}\n', f'p_value: {p}')
"""

#IV testing
"""
#attemping tocreating endogenous variable and instrument
instrument = test_data['control_0'].copy()
instrument.name = 'instrument'
for i, item in instrument.iteritems():
    c = np.random.random()
    instrument[i] = instrument[i]*c
inst_endog = pd.DataFrame(
        {
                instrument.name: instrument,
                'control_0': test_data['control_0'],
        }
)
res_reg = Regression(inst_endog)
res_reg.ols('control_0', ['instrument',])
res = res_reg.res
print(res[0:2,0])
instrument = pd.Series(list(res.flat), name = 'instrument')
print(inst_endog[0:2])
inst_endog.drop(columns = 'instrument', inplace = True)
inst_endog.insert(0, instrument.name, instrument)
print(inst_endog[0:2])
inst_endog_reg = Regression(inst_endog)
inst_endog_reg.ols('control_0', ['instrument',])
print('\ninstrument against endog\n')
print(inst_endog_reg.coef)

inst_out = pd.DataFrame(
        {
                instrument.name: instrument,
                'outcome': test_data['outcome'],
        }
)
inst_out = Regression(inst_out)
inst_out.ols('outcome', ['instrument',])
print('\ninstrument against outcomes\n')
print(inst_out.coef)
test_data.insert(0, instrument.name, instrument)
"""
"""
#General IV testing
iv = Regression(test_data)
iv.iv(
      'outcome',
      control_vars,
      endog = 'control_0',
      instrument = 'control_1',
      intercept = True,
)
print('\nIV\n',iv.coef, '\nfirst stage\n', iv.fs_coef, '\n', iv.se)
print(iv.fs_f_stat)
"""

#Group Testing
"""
#insert grouping variable if necessary
groups = [round(i, -2) for i in range(1, test_data.shape[0]+1)]
test_data.insert(0, 'groups', groups)
"""
"""
cluster_reg = Regression(test_data)
cluster_reg.cluster_ols(
        'outcome',
        control_vars,
        grouping_var = 'groups',
        intercept = True)
print('\ncluster\n', cluster_reg.coef)
print(cluster_reg.cluster_se)
F, p = cluster_reg.f_test(
        test_type = 'anova',
        var_to_analyze = 'control_0',
        grouping_var = 'groups'
)
print(f'\nF-stat: {F}\n', f'p_value: {p}')
"""
"""
cluster_iv = Regression(test_data)
cluster_iv.cluster_iv(
        'outcome', 
        control_vars, 
        endog = 'control_0', 
        instrument = 'control_1', 
        grouping_var = 'groups', 
        intercept = True,
)
print('\ncluster\n', cluster_iv.coef)
print(cluster_iv.cluster_se, '\n', cluster_iv.se)
"""
