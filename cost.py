
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
    def __init__(self,name,x_t):
        self.name = name
       
        self.Q = np.identity(2) * 1
        self.Q = 0.5 * self.Q # 0.1
        self.Q[1,1] = 0
        
        self.R = 1 * np.identity(2) * 1
        
        self.ix = 3
        self.iu = 2
        self.x_t = x_t
        
        # self.x_des = des;
        # self.setGoal(des)
        
    def setGoal(self,des) :
        
        self.goal = np.zeros(3)
        self.goal[0] = des[0]
        self.goal[1] = des[1]
        self.goal[2] = des[2]
        
    def estimateCost(self,x,u):
        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
            
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

        
        return cost_total
    
    def diffCost(self,x,u):
        
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1

        else :
            N = np.size(x,axis = 0)

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
        c_nominal = self.estimateCost(x,u)
        c_change = self.estimateCost(x_aug,u_aug)
        c_change = np.reshape(c_change,(N,1,iu+ix))


        c_diff = ( c_change - np.reshape(c_nominal,(N,1,1)) ) / h
        c_diff = np.reshape(c_diff,(N,iu+ix))
            
        return  np.squeeze(c_diff)
    
    def hessCost(self,x,u):
        
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1

        else :
            N = np.size(x,axis = 0)
        
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
        c_nominal = self.diffCost(x,u)
        c_change = self.diffCost(x_aug,u_aug)
        c_change = np.reshape(c_change,(N,iu+ix,iu+ix))
        c_hess = ( c_change - np.reshape(c_nominal,(N,1,ix+iu)) ) / h
        c_hess = np.reshape(c_hess,(N,iu+ix,iu+ix))
         
        return np.squeeze(c_hess)
