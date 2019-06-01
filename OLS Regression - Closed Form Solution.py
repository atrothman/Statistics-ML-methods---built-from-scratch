
# coding: utf-8

# # Ordinary Least Squares (OLS) Regression - Closed Form Solution

# Here we will explore Ordinary Least Squares (OLS) Regression in closed from where we will: <br>
# * Specify the statistical model and it's functional form
# * Using the Iris dataset as a motivating example, we will recover estimates of the parameters of the model we specified in 

# ## 1) Model Specification & Analytic Solution in Closed Form
# ### Variables & Dimensions
# Let us begin by specifying our variable relations and matrix dimensions:
# 
# $$
# \begin{align}
# Y &= X \beta + \epsilon \\
# Y &= X \hat{\beta} + e \\
# \hat{Y} &= X\hat{\beta} \\
# \end{align}
# $$
# 
# $\begin{aligned}
# n=
# \end{aligned}$ number of observations
# 
# $\begin{aligned}
# p=
# \end{aligned}$ number of parameters
# 
# Note, that in this formulation we are imbedding the intercept term into the Design and Parameter matrices
# 
# $$
# \begin{align}
# Y &: n\times 1 \\
# X &: n\times (p+1) \\
# \beta &: (p+1)\times 1 \\
# \epsilon, e &: n\times 1
# \end{align}
# $$
# 
# ### Cost Function (Residual Sum of Squares)
# 
# $$
# \begin{align}
# RSS &= e^{T}e^{} \\
# &= (Y - X \hat{\beta})^{T}(Y - X \hat{\beta}) \\
# &= (Y^{T} - \hat{\beta}^{T}X^{T})(Y - X \hat{\beta}) \\
# &= Y^{T}Y -Y^{T}X\hat{\beta} - \hat{\beta}^{T}X^{T}Y + \hat{\beta}^{T}X^{T}X\hat{\beta} \\
# &= Y^{T}Y - 2\hat{\beta}^{T}X^{T}Y + \hat{\beta}^{T}X^{T}X\hat{\beta} \\
# \end{align}
# $$
# 
# ### First and Second Partial Derivatives (with respect to Beta)
# 
# $$
# \begin{align}
# \frac{\partial RSS}{\partial \hat{\beta}} &= -2X^{T}Y + 2X^{T}X \hat{\beta} \\
# \frac{\partial^{2} RSS}{\partial^{2} \hat{\beta}} &= 2X^{T}X \\
# \end{align}
# $$
# 
# ### Analytic Solution in Closed Form 
# Let us recover estimates of Beta that minimize the cost function analytically. We can do so by setting the first partial derivative to 0, and solving for Beta:
# 
# $$
# \begin{align}
# \frac{\partial RSS}{\partial \hat{\beta}} = -2X^{T}Y + 2X^{T}X \hat{\beta} &\stackrel{set}{=} 0 \\
# 2X^{T}X \hat{\beta} &= 2X^{T}Y \\
# X^{T}X \hat{\beta} &= X^{T}Y \\
# \hat{\beta} &= (X^{T}X)^{-1} X^{T}Y \\
# \end{align}
# $$

# ## 2) Motivating Example with the "Iris" Dataset
# We will show the above closed form solution in action with a motivating example. We will use the Iris Dataset to do so by:
# * Using "Sepal Length" as our outcome of interest, with all remaining variables as covariates in the regression model
#  * Note, the variable "Species" is reparameterized as "one-hot" coding, with the category "virginica" set as the reference category

# In[13]:


######################
## import libraries ##
######################
import numpy as np
import pandas as pd
from sklearn import datasets
import statsmodels.formula.api as sm
from numpy.linalg import inv


# In[14]:


###################################
## import and clean iris dataset ##
###################################
iris = datasets.load_iris()
df = pd.DataFrame()
df['S_Length'] = iris['data'][:,0]
df['Intercept']=np.full(iris['data'].shape[0], 1)
df['S_Width'] = iris['data'][:,1]
df['P_Length'] = iris['data'][:,2]
df['P_Width'] = iris['data'][:,3]
df['Species'] = iris['target']
df['Species'] = df['Species'].apply(str)
df.loc[df['Species']==str(0), "Species"] = str(iris['target_names'][0])
df.loc[df['Species']==str(1), "Species"] = str(iris['target_names'][1])
df.loc[df['Species']==str(2), "Species"] = str(iris['target_names'][2])
df['Species_setosa']=0
df.loc[df['Species']=='setosa', 'Species_setosa']=1
df['Species_versicolor']=0
df.loc[df['Species']=='versicolor', 'Species_versicolor']=1
df = df.drop('Species', axis=1)
df.describe()


# In[15]:


## creat arrays for closed form solution
Y = np.array(df['S_Length']).reshape((len(df['S_Length']), 1))
X = np.array(df[['Intercept', 'S_Width', 'P_Length', 'P_Width', 'Species_setosa', 'Species_versicolor']])


# ## 3) Compute Solution

# In[16]:


## Beta estimates
Beta_estimate = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), Y)
print(Beta_estimate)


# Let's compare this solution to that provided by the OLS model provided in the "statsmodels" package

# In[17]:


## built in package
results = sm.ols(formula="S_Length ~ S_Width + P_Length + P_Width + Species_setosa + Species_versicolor", data=df).fit()
print(results.params)

