# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:19:25 2019

@author: Andrew
"""

######################
## import libraries ##
######################
from scipy.misc import factorial
import numpy as np
import pandas as pd
from sklearn import datasets
import statsmodels.formula.api as sm
from statsmodels.genmod.families import Poisson
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
LR_df.loc[LR_df['S_Length']<5.1, 'S_Length'] = 0
LR_df.loc[(LR_df['S_Length']>=5.1) & (LR_df['S_Length']<5.8), 'S_Length'] = 1
LR_df.loc[(LR_df['S_Length']>=5.8) & (LR_df['S_Length']<6.4), 'S_Length'] = 2
LR_df.loc[LR_df['S_Length']>=6.4, 'S_Length'] = 3
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
A = np.exp(Z)
ones_vector = np.full(Y.shape[0], 1).reshape(Y.shape[0], 1)

######################
## Gradient Descent ##
######################
m = 50000
alpha = 0.0002
J = pd.DataFrame()
J['iterative_step'] = range(0,m+1)
J['cost'] = np.full(m+1, None)
J.loc[0, 'cost'] = np.asscalar(-np.dot(Y.T, np.dot(X, Beta)) + np.dot((A+factorial(Y)).T, ones_vector))                        

for i in range(1, m+1):    
    J_partial_Beta = np.dot(X.T, (A-Y))
    Beta = Beta - (alpha*J_partial_Beta)
    Z = np.dot(X, Beta)
    A = np.exp(Z)
    J.loc[i, 'cost'] = np.asscalar(-np.dot(Y.T, np.dot(X, Beta)) + np.dot((A+factorial(Y)).T, ones_vector))   
    del J_partial_Beta    

plt.plot(J['iterative_step'], J['cost'])
plt.title('Gradient Descent') 
plt.xlabel('Iterative Step') 
plt.ylabel('Cost') 
Beta

## built in package
results = sm.glm(formula="S_Length ~ S_Width + P_Length + P_Width + Species_setosa + Species_versicolor", data=LR_df, family=Poisson()).fit()
print(results.params)


