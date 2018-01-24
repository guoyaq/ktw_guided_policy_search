
# coding: utf-8

# In[1]:

from __future__ import division
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
import tensorflow as tf
from termcolor import colored
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))


# In[ ]:

class policy :
    def __init__(self,name,ix,iu,N) :
        self.name = name;
        self.ix = ix
        self.iu = iu
        self.N = N
        
        # feedback and forward gain
        self.K_mat = np.zeros((N,iu,ix))
        self.k_mat = np.zeros((N,iu))
        
        # nominal state
        self.x_nominal = np.zeros((N,ix))
        self.u_nominal = np.zeros((N,iu))
        
        # policy variance
        self.polVar = np.zeros((N,iu,iu))
        
    def setter(self,K_mat,k_mat,x,u,var) :
        self.K_mat = K_mat
        self.k_mat = k_mat
        self.x_nominal = x
        self.u_nominal = u
        self.polVar = var
        
        

