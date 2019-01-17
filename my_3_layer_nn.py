# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:53:25 2019

@author: cks
"""

import numpy as np

def my_sig(x,deriv=False):
    if(deriv==True):
            return x*(1-x)
    return 1/(1+np.exp(-x))

# Initialize the input array
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
    
#Initialize the output array
y = np.array([[0],
              [1],
              [1],
              [0]])
    
#Seed the random number generator so as to obtain the same random number distribution everytime the code is run
np.random.seed(1)

#Randomly Initialize the W1 and W2 weight matrices that link input to layer 1 and the outputs of layer 1 to layer 2 respectively
w1 = 2*np.random.random((3,4)) - 1
w2 = 2*np.random.random((4,1)) - 1

for i in range(50000):
    
    #Feedforward 
    l0 = X
    l1 = my_sig(np.dot(l0,w1))
    l2 = my_sig(np.dot(l1,w2))
    
    #Error for layer 2 with actual output
    l2_error = y-l2
    
    if(i%10000 == 0):
        print("Error : ", np.mean(np.abs(l2_error))) 
    
    #Amount by which we have to tweak the weights with respect to the gradient of the error
    l2_update = l2_error*my_sig(l2,deriv=True)
    
    # Error and update for layer 1
    l1_error = np.dot(l2_update,w2.T)    
    
    l1_update = l1_error * my_sig(l1,deriv=True)
    
    #Weight Adjustment 
    w2 += np.dot(l1.T,l2_update)
    w1 += np.dot(l0.T,l1_update)