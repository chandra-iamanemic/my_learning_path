# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:53:17 2019

@author: Chandrasekar Sivaraman
"""

#Creating a function to do the zero padding operation
import numpy as np

def my_zero_pad(X, pad_value):
    # Assuming that the first dimension of the input matrix is the number of samples in the data set
    #2nd and 3rd dimension of the array are our pixel dimensions and 4th dimension is the number of channels
    #We are padding zeros only to the 2nd and 3rd dimensions and not to the channels
    
    padded_mat = np.pad(X, ((0, 0), (pad_value,pad_value), (pad_value, pad_value), (0, 0)), 'constant', constant_values=0)

    return padded_mat


def my_conv(input_layer, filters, bias, pad, stride):
    
    #m = number of samples in the set
    #n_h_inp = height of input layer
    #n_w_inp = width of input layer
    #n_c_inp = channels of input layer
    (m, n_h_inp, n_w_inp, n_c_inp) = input_layer.shape
    
    #f = fitler height and width 
    #note that n_c_inp should match since the convolution is performed across the channels
    #n_f = number of filters
    (f,f,n_c_inp,n_f) = filters.shape
    
    
    #Number of values after convolution
    #Height and Width are given by ((n-f+2p)/s) + 1
    
    n_h = int(((n_h_inp - f + 2*pad)/stride)) + 1 
    n_w = int(((n_w_inp - f + 2*pad)/stride)) + 1 
    
    #The feature matrix after the convolution
    #it will have a 3d volume of height n_h, width n_w and depth equal to the number of filters used n_f
    #since we have m examples we have m of such 3d volumes
    feature_mat = np.zeros((m,n_h,n_w,n_f))
    
    #Zero pad the input before convolving to preserve the dimensions
    input_layer_padded = my_zero_pad(input_layer,pad)

    #Lets begin the convolution process
    
    #Loop over all the samples in input space
    for l in range(m):       
        #Select the current sample to convolve                    
        current_inp = input_layer_padded[l] 
        
        #Nested for loops to cover the 3D space of the output feature matrix for one sample (n_h, n_w, n_f)
        for i in range(n_h):
            for j in range(n_w):
                for k in range(n_f):
                    
                    #pick out the current volume slice that you want to convolve
                    current_slice = current_inp[(i*stride):(i*stride)+f,(j*stride):(j*stride)+f, : ]
                    current_filter = filters[:,:,:,k]
                    current_bias = bias[:,:,:,k]
                    current_conv = (np.multiply(current_slice,current_filter)) + current_bias
                    
                    feature_mat[l,i,j,k] = np.sum(current_conv)
                    
                
        
    cache = (input_layer, filters, bias, stride, pad)
    
    return feature_mat, cache









def my_maxpool(input_layer,f,stride):
    
    #m = number of samples in the set
    #n_h_inp = height of input layer
    #n_w_inp = width of input layer
    #n_c_inp = channels of input layer

    (m, n_h_inp, n_w_inp, n_c_inp) = input_layer.shape
    
    n_h = int(((n_h_inp - f)/stride)) + 1 
    n_w = int(((n_w_inp - f)/stride)) + 1 
    n_c = n_c_inp
    
    output_mat = np.zeros((m,n_h,n_w,n_c))
    
    for l in range(m):       
        
        #Nested for loops to cover the 3D space of the output feature matrix for one sample (n_h, n_w, n_f)
        for i in range(n_h):
            for j in range(n_w):
                for k in range(n_c):
    
                    current_slice = input_layer[l,(i*stride):(i*stride)+f,(j*stride):(j*stride)+f, k]
    
                    output_mat[l,i,j,k] = np.max(current_slice)
                
                    cache = (input_layer,f,stride)
                
    return output_mat,cache
 
    




    
def my_backprop(dfeature_mat,cache):
    
    #Get your required information from the cached values
    
    (input_layer, filters, bias, stride, pad) = cache
    
    #m = number of samples in the set
    #n_h_inp = height of input layer
    #n_w_inp = width of input layer
    #n_c_inp = channels of input layer
    (m, n_h_inp, n_w_inp, n_c_inp) = input_layer.shape
    
    #f = fitler height and width 
    #note that n_c_inp should match since the convolution is performed across the channels
    #n_f = number of filters
    (f,f,n_c_inp,n_f) = filters.shape
    
    #Determine the shape values of the gradient matrix which is input to the function
    #dfeature_mat is the gradient of the loss w.r.t feature_mat
    (m,n_h,n_w,n_c) = dfeature_mat.shape
    
    #Note that n_c will be equal to n_f since the output channels is equal to number of filters during convoltution
                    
    
    #dinput_layer is the gradient of the loss w.r.t input_layer
    #dfilters is the gradient of the loss w.r.t filters
    #dbias is the gradient of the loss w.r.t bias
    dinput_layer =  np.zeros((m, n_h_inp, n_w_inp, n_c_inp))
    dfilters = np.zeros((f,f,n_c_inp,n_f))
    dbias = np.zeros((1,1,1,n_c))
    
    #pad the input_layer and dinput_layer
    input_layer_padded = my_zero_pad(input_layer,pad)
    dinput_layer_padded = my_zero_pad(dinput_layer,pad)
    
    
    for l in range(m):
        
        current_input_layer_padded = input_layer_padded[l]
        current_dinput_layer_padded = dinput_layer_padded[l]
        
        for i in range(n_h):
            for j in range(n_w):
                
                for k in range(n_c):               
                    current_input_slice = current_input_layer_padded[(i*stride):(i*stride)+f,(j*stride):(j*stride)+f, :]
                    current_dinput_slice = current_dinput_layer_padded[(i*stride):(i*stride)+f,(j*stride):(j*stride)+f, :]
                    
                    # feature = input * filter
                    # if error depends on f and its gradients wrt to feature map is df
                    # then the gradient of error wrt input is df*filters by chain rule
                    # and the gradient of error wrt filters is df*input by chain rule
                    
                    current_dinput_slice += filters[:,:,:,k]*dfeature_mat[l,i,j,k]
                    dfilters[:,:,:,k] += current_input_slice*dfeature_mat[l,i,j,k]
                    dbias[:,:,:,k] += dfeature_mat[l,i,j,k]
        
        
        dinput_layer[l,:,:,:] = current_dinput_layer_padded[pad:-pad, pad:-pad, :]
        
        
    
    return dinput_layer, dfilters, dbias


def my_maxpool_tracker(x):
    
    maxpool_tracker = (x==np.max(x))
    
    return maxpool_tracker


def my_maxpool_backprop(doutput_mat,cache):
    
    (input_layer,f,stride) =  cache
   
    (m, n_h_inp, n_w_inp, n_c_inp) = input_layer.shape
   
    (m, n_h, n_w, n_c) = doutput_mat.shape
   
    dinput_layer = np.zeros(input_layer.shape)

    
    for l in range(m):
        
        current_inp = input_layer[l]
        
        for i in range(n_h):
            for j in range(n_w):
                for k in range(n_c):
                    
                    current_input_slice = current_inp[(i*stride):(i*stride)+f,(j*stride):(j*stride)+f, k]
                    
                    #track the positions of the maximum values extracted by maxpool
                    tracker = my_maxpool_tracker(current_input_slice)
                    
                    #assign the gradients only to the values that contributed to the maximum value 
                    dinput_layer[l,(i*stride):(i*stride)+f,(j*stride):(j*stride)+f, k] += np.multiply(tracker,doutput_mat[l,i,j,k])
                    
                    
    return dinput_layer





import cv2

angry_cat = cv2.imread("C:/Users/cks/Documents/practice codes/angry cat.jpg")
angry_cat = cv2.resize(angry_cat, (256,256))

angry_cat = np.reshape(angry_cat,[1,256,256,3])

cat_img = angry_cat[0,:,:,:]

cv2.imshow("resized cat img",cat_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

random_filter = 2*np.random.rand(3,3,3) - 1

random_filter = np.reshape(random_filter,[1,3,3,3])
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)

z, cache = my_conv(angry_cat, W, b, 0,1)

for i in range(8):
    cv2.imshow("resized cat img",z[0,:,:,i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()








        
        

        
        
        