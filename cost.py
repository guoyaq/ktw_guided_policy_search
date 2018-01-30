
# coding: utf-8

# In[1]:

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
from gps_util import getCostNN, getCostAppNN, getCostTraj, getObs
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))


# In[71]:

class tracking :
    def __init__(self,name,x_t,N,flag_const):
        self.name = name
       
        self.Q = np.identity(2) * 1
        self.Q = 0.5 * self.Q # 0.1
        self.Q[1,1] = 0
        
        self.R = 1 * np.identity(2) * 1
        
        self.ix = 3
        self.iu = 2
        self.x_t = x_t
        self.ic = 1
        self.N = N
        self.flag_const = flag_const
        
    def ineqConst(self,x,u) :
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        c1 = np.expand_dims(- x[:,0] + x[:,1],axis=1)
        # c2 = np.expand_dims(u[:,0] - 1.0,axis=1)
        # c = np.hstack((c1,c2))
        c = c1

        return c
        
    def estimateCost(self,x,u,Mu=None,lam=None):    
        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            if Mu is None :
                Mu = np.zeros((N,self.ic,self.ic))
                lam = np.zeros((N,self.ic,self.ic))
            else :
                Mu = np.expand_dims(Mu,axis=0) # N * ic * ic
                lam = np.expand_dims(lam,axis=0) # N * ic * ic
        else :
            N = np.size(x,axis = 0)
            if Mu is None :
                Mu = np.zeros((N,self.ic,self.ic))
                lam = np.zeros((N,self.ic,self.ic))
            
        # distance to target      
        d = np.sqrt( np.sum(np.square(x[:,0:2] - self.x_t),1) )
        d = np.expand_dims(d,1)
        d_1 = d - 0
        
        # theta diff
        y_diff = np.zeros((N,1))
        x_diff = np.zeros((N,1))
        y_diff[:,0] = self.x_t[1] - x[:,1]
        x_diff[:,0] = self.x_t[0] - x[:,0]
        
        theta_target = np.arctan2(y_diff,x_diff)
        theta_diff = np.expand_dims(x[:,2],1) - theta_target

        x_mat = np.expand_dims(np.hstack((d_1,theta_diff)),2)
        Q_mat = np.tile(self.Q,(N,1,1))
        
        lx = np.squeeze( np.matmul(np.matmul(np.transpose(x_mat,(0,2,1)),Q_mat),x_mat) )
        
        # cost for input
        u_mat = np.expand_dims(u,axis=2)
        R_mat = np.tile(self.R,(N,1,1))
        lu = np.squeeze( np.matmul(np.matmul(np.transpose(u_mat,(0,2,1)),R_mat),u_mat) )

        cost_total = 0.5*(lx+lu)

        # inequality constraint
        ineq_c = self.ineqConst(x,u)
        ineq_c_mat = np.expand_dims(ineq_c,axis=2)

        # Mu matrix setting
        flag_c = np.logical_or( ineq_c > 0, np.diagonal(lam,0,2,1) > 0 )
        Mu = np.expand_dims(flag_c,2) * Mu
        # print Mu
        # cost for constraint
        cost_mu = 0.5 * np.squeeze(np.matmul(np.matmul(np.transpose(ineq_c_mat,(0,2,1)),Mu),ineq_c_mat))
        cost_lam = np.squeeze( np.matmul( np.transpose( np.expand_dims(np.diagonal(lam,0,2,1),2),(0,2,1)) , ineq_c_mat ) )
        cost_const = ( cost_mu + cost_lam ) * self.flag_const
        
        return (cost_total + cost_const)
    
    def diffCost(self,x,u,Mu=None,lam=None):    
        # dimension
        ndim = np.ndim(x)
        flagDim = False
        if ndim == 1: # 1 step state & input
            N = 1
            flagDim = True
            if Mu is None :
                Mu = np.zeros((self.ic,self.ic))
                lam = np.zeros((self.ic,self.ic))
        else :
            N = np.size(x,axis = 0)
            if Mu is None :
                Mu = np.zeros((N,self.ic,self.ic))
                lam = np.zeros((N,self.ic,self.ic))

        # state & input size
        ix = self.ix
        iu = self.iu        

        # numerical difference
        h = pow(2,-17)
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_diag x], [u, u_diag]
        x_aug = x_diag + eps_x * h
        x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
        x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))
        
        u_aug = u_diag + eps_u * h
        u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
        u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))


        # numerical difference
        c_nominal = self.estimateCost(x,u,Mu,lam)
        if flagDim == True :
            Mu_temp = np.repeat(np.expand_dims(Mu,axis=0),iu+ix,0)
            lam_temp = np.repeat(np.expand_dims(lam,axis=0),iu+ix,0)
        else :
            Mu_temp = np.repeat(Mu,iu+ix,0)
            lam_temp = np.repeat(lam,iu+ix,0)            
        # print_np(x_aug)
        # print x_aug
        # print_np(Mu_temp)
        c_change = self.estimateCost(x_aug,u_aug,Mu_temp,lam_temp)
        # print_np(c_change)
        c_change = np.reshape(c_change,(N,1,iu+ix))


        c_diff = ( c_change - np.reshape(c_nominal,(N,1,1)) ) / h
        c_diff = np.reshape(c_diff,(N,iu+ix))
            
        return  np.squeeze(c_diff)
    
    def hessCost(self,x,u,Mu=None,lam=None):    
        # dimension
        ndim = np.ndim(x)
        flagDim = False
        if ndim == 1: # 1 step state & input
            N = 1
            flagDim = True
            if Mu is None :
                Mu = np.zeros((self.ic,self.ic))
                lam = np.zeros((self.ic,self.ic))
        else :
            N = np.size(x,axis = 0)
            if Mu is None :
                Mu = np.zeros((N,self.ic,self.ic))
                lam = np.zeros((N,self.ic,self.ic))

        # state & input size
        ix = self.ix
        iu = self.iu
        
        # numerical difference
        h = pow(2,-17)
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_diag x], [u, u_diag]
        x_aug = x_diag + eps_x * h
        x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
        x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))

        u_aug = u_diag + eps_u * h
        u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
        u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))


        # numerical difference
        c_nominal = self.diffCost(x,u,Mu,lam)
        if flagDim == True :
            Mu_temp = np.repeat(np.expand_dims(Mu,axis=0),iu+ix,0)
            lam_temp = np.repeat(np.expand_dims(lam,axis=0),iu+ix,0)
        else :
            Mu_temp = np.repeat(Mu,iu+ix,0)
            lam_temp = np.repeat(lam,iu+ix,0) 
        c_change = self.diffCost(x_aug,u_aug,Mu_temp,lam_temp)
        c_change = np.reshape(c_change,(N,iu+ix,iu+ix))
        c_hess = ( c_change - np.reshape(c_nominal,(N,1,ix+iu)) ) / h
        c_hess = np.reshape(c_hess,(N,iu+ix,iu+ix))
         
        return np.squeeze(c_hess)
