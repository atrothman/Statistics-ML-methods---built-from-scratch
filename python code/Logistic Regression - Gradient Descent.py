# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:19:25 2019

@author: Andrew
"""

######################
## import libraries ##
######################
import numpy as np
import pandas as pd
from sklearn import datasets
import statsmodels.formula.api as sm
from statsmodels.genmod.families import Binomial
from numpy.linalg import inv
import matplotlib.pyplot as plt 

###################################
## import and clean iris dataset ##
###################################
iris = datasets.load_iris()
LR_df = pd.DataFrame()
LR_df['S_Length'] = iris['data'][:,0]
LR_df['Intercept']=np.full(iris['data'].shape[0], 1)
LR_df['S_Width'] = iris['data'][:,1]
LR_df['P_Length'] = iris['data'][:,2]
LR_df['P_Width'] = iris['data'][:,3]
LR_df['Species'] = iris['target']
LR_df['Species'] = LR_df['Species'].apply(str)
LR_df.loc[LR_df['Species']==str(0), "Species"] = str(iris['target_names'][0])
LR_df.loc[LR_df['Species']==str(1), "Species"] = str(iris['target_names'][1])
LR_df.loc[LR_df['Species']==str(2), "Species"] = str(iris['target_names'][2])
LR_df['Species_setosa']=0
LR_df.loc[LR_df['Species']=='setosa', 'Species_setosa']=1
LR_df['Species_versicolor']=0
LR_df.loc[LR_df['Species']=='versicolor', 'Species_versicolor']=1
LR_df.loc[LR_df['S_Length']<5.8, 'S_Length'] = 0
LR_df.loc[LR_df['S_Length']>=5.8, 'S_Length'] = 1
LR_df = LR_df.drop('Species', axis=1)

#######################################
## creat arrays for Gradient Descent ##
#######################################
Y = np.array(LR_df['S_Length']).reshape((len(LR_df['S_Length']), 1))
X = np.array(LR_df[['Intercept', 'S_Width', 'P_Length', 'P_Width', 'Species_setosa', 'Species_versicolor']])

#################################
## initialize parameter matrix ##
#################################
k = X.shape[1]
np.random.seed(10815657)
nudge=0.1
Beta = np.random.uniform(low=-1*nudge, high=1*nudge, size=k).reshape(k, 1)
Z = np.dot(X, Beta)
A = 1 / (1+np.exp(-Z))

######################
## Gradient Descent ##
######################
m = 100000
alpha = 0.001
J = pd.DataFrame()
J['iterative_step'] = range(0,m+1)
J['cost'] = np.full(m+1, None)
J.loc[0, 'cost'] = np.asscalar(-np.dot(Y.T, np.log(A)) -np.dot((1-Y.T), np.log(1-A)))                             

for i in range(1, m+1):    
    J_partial_Beta = np.dot(X.T, (A-Y))
    Beta = Beta - (alpha*J_partial_Beta)
    Z = np.dot(X, Beta)
    A = 1 / (1+np.exp(-Z))
    J.loc[i, 'cost'] = np.asscalar(-np.dot(Y.T, np.log(A)) -np.dot((1-Y.T), np.log(1-A))) 
    del J_partial_Beta    

plt.plot(J['iterative_step'], J['cost'])
plt.title('Gradient Descent') 
plt.xlabel('Iterative Step') 
plt.ylabel('Cost') 
Beta

## built in package
results = sm.glm(formula="S_Length ~ S_Width + P_Length + P_Width + Species_setosa + Species_versicolor", data=LR_df, family=Binomial()).fit()
print(results.params)

