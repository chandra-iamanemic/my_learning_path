# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:28:03 2019

@author: cks
"""

#Import the required Libraries
import time
import cv2

#Initiate the video capture using the webcam
vidcap = cv2.VideoCapture(0)
count = 0
#Run a loop to capture desired frames/images from a motion captured by the webcam
#Press a key to initate a frame capture
while True:
    success,image = vidcap.read()
    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.imwrite("C:/Users/cks/Documents/practice codes/frames/frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1 
    if count>20:
        break

