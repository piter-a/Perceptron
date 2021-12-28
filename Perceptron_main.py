#!/usr/bin/env python
# coding: utf-8

# In[116]:


#https://github.com/yacineMahdid/artificial-intelligence-and-machine-learning/blob/master/deep-learning-from-scratch-python/perceptron.ipynb
import numpy as np
import random
import pandas as pd
import csv, collections
import matplotlib.pyplot as plt
from random import randrange

# Perceptron algorithm
class Perceptron():
    
    def __init__(self):
        self.weights = []
        
    # Fitting the learning data
    # Parameteres:
    # X= Teaching vectors of dimensions: [number of examples, number of features]
    # y= dimension [number of examples], target values
    # The fit function introduces weights in the self.weights object to the vector with number of features+1 
    # learning_rate: Used to limit the amount each weight is corrected each time it is updated.
    def fit(self, X, y, learning_rate = 0.01, n_iter = 1000):
        
        (num_row, n_feature) = X.shape
        
        # # weights vector contains small randomly generated numbers as shown below
        # The weigths vector is not initialized to 0, as learning facotr impacts results of classification only when the initial
        # values are >0 .
        self.weights = np.random.rand(n_feature+1) 

        # starts the training algorithm
        for i in range(n_iter):
            
            # Stochastic Gradient Descent
            rand = random.randint(0,num_row-1)
            row = X[rand,:] # random sample from the dataset
            yhat = self.predict(row)
            error = (y[rand] - yhat) # gradient estimation
            self.weights[0] = self.weights[0] + learning_rate*error*1 # first weight one is the bias

            # Update all parameters after bias
            for f_i in range(n_feature):
                self.weights[f_i] = self.weights[f_i] + learning_rate*error*row[f_i]
                
            if i % 100 == 0: #step size at which mean error shown
                total_error = 0
                for rand in range(num_row):
                    row = X[rand,:]
                    yhat = self.predict(row)
                    error = (y[rand] - yhat)
                    total_error = total_error + error**2
                mean_error = total_error/num_row
                print(f"Iteration {i} with error = {mean_error}")
    #predicts an output value for a row given a set of weights.   
    def predict(self, row):
            
        # Starts with the bias at weights == 0
        activation = self.weights[0]
        
        # Iterating over the weights and the features vector in each row
        for weight, feature in zip(self.weights[1:], row):
            activation = activation + weight*feature
            
        #  Step Function 
        if activation >= 0.0:
            return 1.0
        return 0.0







# Loading data
filename= open(r"C:\Users\23gab\OneDrive\Desktop\train.data")
df = pd.read_csv(filename, header=None, encoding='utf-8')



# Binary perceptron to distinguish between class 1 and class 2 only
def class1_2():
    y = df.iloc[0:80, 4].values
    y = np.where(y == 'class-1', -1, 1)
    y= np.array(y)
    # selection of class 1 and 2 only (first 80 values from data file) and four features
    X = df.iloc[0:80, [0,1,2,3]].values
    X= np.array(X)
    
    
    # Sorting the X and y data 
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    ppt = Perceptron()
    ppt.fit(X,y, n_iter = 1000)
    
    # Creating y1 and x1 values to use on a 2d plot (to visualize and see if data is linearly separable)
    #(uses only 3rd and 4th features (e.g [2,3]) as there are biggest differences between them for all classes)
    y1 = df.iloc[0:80, 4].values
    y1 = np.where(y == 'class-1', -1, 1)
    # extract 3rd and 4th features
    X1 = df.iloc[0:80, [2,3]].values
    # plot data
    plt.scatter(X1[:40, 0], X1[:40, 1],
            color='red', marker='o', label='class-1')
    plt.scatter(X1[40:80, 0], X1[40:80, 1],
            color='blue', marker='x', label='class-2')
    plt.xlabel('feature 3')
    plt.ylabel('feature 4')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    
    
    


#Binary perceptron to distinguish between class 2 and class 3 only
def class2_3():
    
    # selection of class 2 and 3 only (last 80 values from data file) and four features
    y = df.iloc[40:120, 4].values
    y = np.where(y == 'class-2', -1, 1)
    y= np.array(y)
    X = df.iloc[40:120, [0,1,2,3]].values
    X= np.array(X)
    
    # Sorting the X and y data
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    ppt = Perceptron()
    ppt.fit(X,y, n_iter = 1000)
    
    #Creating y1 and x1 values to use on a 2d plot (to visualize and see if data is linearly separable)
    #(uses only 3rd and 4th features (e.g [2,3]) as there are biggest differences between them for all classes)
    X1 = df.iloc[40:120, [2,3]].values
    # extract 3rd and 4th features
    y1 = df.iloc[40:120, 4].values
    y1 = np.where(y == 'class-1', -1, 1)
    
    # plotting data
    plt.scatter(X1[:40, 0], X1[:40, 1],color='red', marker='o', label='class-2')
    plt.scatter(X1[40:80, 0], X1[40:80, 1],color='blue', marker='x', label='class-3')
    plt.xlabel('feature 3')
    plt.ylabel('feature 4')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    



# Binary perceptron to distinguish between class 1 and class 3 only
def class1_3():
    # selection of class 1 and 3 only (first 40 and last 40 values from data file) and four features
    y=df.iloc[np.r_[0:40,80:120], 4].values
    y = np.where(y == 'class-1', -1, 1)
    y= np.array(y)
    X = df.iloc[np.r_[0:40,80:120], [0,1,2,3]].values
    X= np.array(X)
    
    # Sorting the X and y data 
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    ppt = Perceptron()
    ppt.fit(X,y, n_iter = 1000)
    
    # Creating y1 and x1 values to use on a 2d plot (to visualize and see if data is linearly separable)
    #(uses only 3rd and 4th features (e.g [2,3]) as there are biggest differences between them for all classes)
    X1= df.iloc[np.r_[0:40,80:120], [2,3]].values
    y1=df.iloc[np.r_[0:40,80:120], 4].values
    y1= np.where(y == 'class-1', -1, 1)

    # plotting data
    plt.scatter(X1[:40, 0], X1[:40, 1],color='red', marker='o', label='class-1')
    plt.scatter(X1[40:80, 0], X1[40:80, 1],color='blue', marker='x', label='class-3')
    plt.xlabel('feature 3')
    plt.ylabel('feature 4')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
   
    




# 1 vs rest 
def class1vsrest():
   
    # all classes choosen
    y = df.iloc[:, 4].values
    #class 1 gets assigned -1, other classes have assigned value of 1
    y = np.where(y == 'class-1', -1, 1)
    y= np.array(y)
    
    #all features used for training and fitting
    X = df.iloc[:, [0,1,2,3]].values
    X= np.array(X)
    
    # Sorting the X and y data
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    ppt = Perceptron()
    ppt.fit(X,y, n_iter = 1000)
    
    # extracting feature 3 and 4 for plotting 2d graph
    X1 = df.iloc[:, [2,3]].values
    y1 = df.iloc[:, 4].values
    y1 = np.where(y == 'class-1', -1, 1)
    #plotting
    plt.scatter(X1[:40, 0], X1[:40, 1],color='red', marker='o', label='class-2')
    plt.scatter(X1[40:80, 0], X1[40:80, 1],color='blue', marker='x', label='class-2')
    plt.scatter(X1[80:120, 0], X1[80:120, 1],color='yellow', marker='*', label='class-3')
    plt.xlabel('feature 3')
    plt.ylabel('feature 4')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


class1_2()
class2_3()   
class1_3()    
class1vsrest()   













# In[ ]:





# In[ ]:




