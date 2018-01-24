
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
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))


# In[2]:

import model
import cost
import iLQR
import poliOpt


# In[3]:

class pathIntOpt:
    def __init__(self,name,horizon,Model,Cost):
        # class name
        self.name = name

        # cost
        self.cost = Cost
           
        # input & output size
        self.ix = self.model.ix
        self.iu = self.model.iu

        # PI2 parameters
        self.verbosity = True
        self.N = horizon
        
        # dual variable
        self.eta = 1 * np.ones(self.N)
        
        # constraint parameter
        self.epsilon = 10
        
        # line-search step size
        self.Alpha = np.power(10,np.linspace(0,-3,4))
        
        # feedforward, feedback gain
        self.l = np.zeros((self.N,self.model.iu))
        self.L = np.zeros((self.N,self.model.iu,self.model.ix))
        
        # input variance
        self.Quu_save = np.zeros((self.N,self.model.iu,self.model.iu)) # inverse of policy variance
        self.Quu_inv_save = np.zeros((self.N,self.model.iu,self.model.iu)) # policy variance
        
    def setEnv(self,eta,epsilon,K_fixed,x_fit,u_fit,num_fit) :
    
        self.eta = eta
        self.epsilon = epsilon
        self.num_fit = num_fit
        
        # line-search step size
        self.Alpha = np.power(10,np.linspace(0,-3,4))
        
        # feedforward, feedback gain
        self.l = np.zeros((self.N,self.model.iu))
        self.L = K_fixed # fix feedback gain
        
        # input variance
        self.Quu_save = np.zeros((self.N,self.model.iu,self.model.iu))
        self.Quu_inv_save = np.zeros((self.N,self.model.iu,self.model.iu))
        
        # samples
        self.x_fit = x_fit
        self.u_fit = u_fit
        
        # S & P
        self.S = np.zeros((self.N,1,self.num_fit))
        self.P = np.zeros((self.N,1,self.num_fit))
        
    
    def getCost(self,x,u) :
        
        u_temp = np.vstack((u,np.zeros((1,self.model.iu))))
        temp_c = self.cost.estimateCost(x,u_temp)
        
        return np.sum( temp_c )
        
        
    def estimate_cost(self,x,u):
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            K_fit = np.expand_dims(K_fit,axis=0)
            k_fit = np.expand_dims(k_fit,axis=0)
            var_inv = np.expand_dims(var_inv,axis=0)
        else :
            N = np.size(x,axis = 0)
                
        # cost function for iLQR / dual variable
        c1 = self.cost.estimateCost(x,u)
        
        return np.squeeze(c1)
    
    def getCostToGo(self,x_fit,u_fit) :
        
        S = np.zeros((self.N,1,self.num_fit))
        P = np.zeros((self.N,1,self.num_fit))
        
        for i in range(self.num_fit) :
            c = self.estimate_cost(self,x_fit[:,:,i],u_fit[:,:,i])
            for j in range(self.N) :
                S[j,:,i] = c[j:]
                
        return S
    
    def getP(self,eta) :
                
        for j in range(self.N) :
            temp = np.sum( np.exp( -1 / eta[j] * S[j,:,:] ) )
            for i in range(self.num_fit) :        
                P[j,:,i] = np.exp( -1 / eta[j] * S[j,:,i] ) / temp
        
        return P
    
    def updateEta(self,S) :
        
        maxIter = 10
        g_new = 1e5
        g_old = 1e5
        eta = self.eta
        for j in range(self.N) :
            for k in range(maxIter) :
                temp = np.sum( np.exp( -1 / self.eta[j] * S[j,:,:] ) )
                temp2 = np.sum( np.exp( -1 / self.eta[j] * S[j,:,:] ) * S[j,:,:] / (eta[j]*eta[j]) )
                g_old = eta[j] * self.epsilon + eta[j] * np.log( 1 / self.N * temp )
                for ii in self.Alpha :
                    temp3 = eta[j] - self.Alpha[ii] * ( self.epsilon + np.log( 1 / self.N * temp )+ eta[j] * ( temp2 / temp ) )
                    g_new = temp3 * self.epsilon + temp3 * np.log( 1 / self.N * temp )
                    if g_new < g_old :
                        break
                
                                                      
                                                       
    
    def iterDGD(self) :
        
        eta_max = 1e6
        eta_min = 1e-6
        u_temp = uIni
        
        self.S = getCostToGo(self.x_fit,self.u_fit)
        self.eta = updateEta(self,self.S)
        self.P = getP(self.eta)
        
        
         


                
        return x_temp, u_temp, Quu_temp, Quu_inv_temp, self.eta, cost, K_temp, k_temp
        
        



