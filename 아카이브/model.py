
# coding: utf-8

# In[ ]:

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))


# In[ ]:    
class unicycle:
    def __init__(self,name):
        self.name = name;
        self.ix = 3
        self.iu = 2
        self.delT = 0.1
        
    def forwardDyn(self,x,u,idx):
     
        # state & input
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        
        v = u[0]
        w = u[1]
        
        # output
        f = np.zeros_like(x)
        f[0] = v * np.cos(x3)
        f[1] = v * np.sin(x3)
        f[2] = w
        
        return (x + f * self.delT)
    
    def diffDyn(self,x,u):

        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(x,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        v = u[:,0]
        w = u[:,1]    
        
        fx = np.zeros((N,self.ix,self.ix))
        fx[:,0,0] = 1.0
        fx[:,0,1] = 0.0
        fx[:,0,2] = - self.delT * v * np.sin(x3)
        fx[:,1,0] = 0.0
        fx[:,1,1] = 1.0
        fx[:,1,2] = self.delT * v * np.cos(x3)
        fx[:,2,0] = 0.0
        fx[:,2,1] = 0.0
        fx[:,2,2] = 1.0
        
        fu = np.zeros((N,self.ix,self.iu))
        fu[:,0,0] = self.delT * np.cos(x3)
        fu[:,0,1] = 0.0
        fu[:,1,0] = self.delT * np.sin(x3)
        fu[:,1,1] = 0.0
        fu[:,2,0] = 0.0
        fu[:,2,1] = self.delT
        
        return np.squeeze(fx) , np.squeeze(fu)
      
class fitLinearModel:
    def __init__(self,name,ix,iu,N):
        
        self.name = name;
        self.ix = ix
        self.iu = iu
        self.N = N
        self.A = np.zeros((N,ix,ix))
        self.B = np.zeros((N,ix,iu))
        self.C = np.zeros((N,ix))
        
    def update(self,x,u) :
        
        ix = self.ix
        iu = self.iu
        
        num_fit = np.size(u,axis=2)
        
        for i in range(self.N) :
            x_next = x[i+1,:,:]
            x_now = x[i,:,:]
            u_now = u[i,:,:]
            colMat = np.vstack((x_now,u_now,np.ones((1,num_fit))))
            rowMat = np.dot(x_next, np.linalg.pinv(colMat))
            self.A[i,:,:] = rowMat[:,0:ix]
            self.B[i,:,:] = rowMat[:,ix:ix+iu]
            self.C[i,:] = rowMat[:,ix+iu]
        
    def forwardDyn(self,x,u,idx):
        
        return np.squeeze( np.dot(self.A[idx,:,:],x) + np.dot(self.B[idx,:,:],u) + self.C[idx,:] )
    
    def diffDyn(self,x,u):

        # actually, this function does not require x or u as input..
        
        return self.A , self.B


    
    
    
    