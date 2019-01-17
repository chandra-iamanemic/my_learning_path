# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:45:20 2019

@author: cks
"""

import tensorflow as tf
import numpy as np

# equation = x^2 - 5*x + 6

x = tf.Variable(0,dtype = tf.float32)
eq = tf.add(tf.add(x**2, tf.multiply(-12.,x)), 36)
train = tf.train.GradientDescentOptimizer(0.01).minimize(eq)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

for i in range(5000):
    session.run(train)
    
print(session.run(x))

