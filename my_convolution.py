# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 23:02:50 2019
@author: Chandrasekar
"""

#Importing numpy and opencv 
import numpy as np
import cv2

#Loading the angry cat image in grayscale
angry_cat = cv2.imread("C:/Users/cks/Documents/practice codes/angry cat.jpg",0)

#Resizing the image into 256x256 and displaying the resized image
angry_cat_resized = cv2.resize(angry_cat, (256,256))
cv2.imshow("resized cat img",angry_cat_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
#explicit vertical edge detecting filters
vertical_edge_filter = np.array([[1,0,-1],
                        [1,0,-1],
                        [1,0,-1]])

#explicit horizontal edge detecting filters
horizontal_edge_filter = np.array([[1,1,1],
                        [0,0,0],
                        [-1,-1,-1]])


#Defining the function to perform convolution by a single filter over an image and to produce the feature map
def my_conv(img, conv_filters):
    
    feature_map = np.zeros((img.shape[0]-conv_filters.shape[0]+1,
                              img.shape[1] - conv_filters.shape[1]+1))
    
    for i in range(feature_map.shape[0]):
        for j in range(feature_map.shape[1]):
            feature_map[i,j] = np.sum(img[i:(i+conv_filters.shape[0]),j:(j+conv_filters.shape[1])]*conv_filters)
    
    return feature_map
        
#applying the vertical edge filter on the cat image
features_vertical = my_conv(angry_cat_resized,vertical_edge_filter)
cv2.imshow("vertical features", features_vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()


#applying the horizontal edge filter on the cat image
features_horizontal = my_conv(angry_cat_resized,horizontal_edge_filter)
cv2.imshow("horizontal features", features_horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Random filter which can turn out to detect any random shape or feature
random_filter = 2*np.random.rand(3,3) - 1

#applying the random filter on the cat image
features_random = my_conv(angry_cat_resized,random_filter)
cv2.imshow("random features", features_random)
cv2.waitKey(0)
cv2.destroyAllWindows()
