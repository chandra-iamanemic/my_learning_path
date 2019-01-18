# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:38:08 2019

@author: cks
"""

# This program is a basic implementation of a neural network without any libraries

# input and output table
#   Input1 Input2   Input3  Output
#   0      0       1        0
#   1      1       1        1
#   1      0       1        1
#   0      1       1        0


# Test Input after training
#   1      0       0        ?


#Importing Libraries
from numpy import *



class my_nn():
    
    def __init__(self):
        
        #Seeding the random generator to obtain the same random numbers everytime
        random.seed(1)                                 
        
                   
        #Creating a layer (3x1) and assigning random weights to the 3 input connections the the output neuron 
        #Random values between -1 to 1
        
        self.my_w1 = 2*random.random((3,1)) - 1
                                    
                                    
                            
    #Defining our sigmoid activation function for the neurons           
    def my_sig(self,x):
        return 1/(1+exp(-x))
    
    
    #Defining our sigmoid derivative function
    def my_sig_deriv(self,x):
        return x*(1-x)
    
    
    def my_nn_train(self,train_inputs,train_outputs,number_of_iterations):
        
        for i in range(number_of_iterations):
            
            #Pass inputs through the weights and evaluate the output
            output = self.my_nn_predict(train_inputs)
            
            #Calculate the error between predicted output and actual output
            error = train_outputs - output
            
            #Calculate the amount of change in the weights to be made by calculating the derivative of the sigmoid of obtained output
            weight_update = dot(train_inputs.T,error*self.my_sig_deriv(output))
            
            #Update the weights
            self.my_w1 += weight_update
            
    def my_nn_predict(self,inputs):
        
        return self.my_sig(dot(inputs,self.my_w1))
            
            

#Create an object of our NN
my_nn = my_nn()
    
print("The initial random weights of W1 (3x1) is :")
print(my_nn.my_w1)
    
#Create the array according to the truth table you want to train
train_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
train_outputs = array([[0,1,1,0]]).T
    
#Train the neural network with the inputs and outputs 
my_nn.my_nn_train(train_inputs,train_outputs,100000)
    
print("Weights W1 after training")
print(my_nn.my_w1)
    
test_input = array([1,0,0])
print("Prediction made on the new input")
print(my_nn.my_nn_predict(test_input))
    


            