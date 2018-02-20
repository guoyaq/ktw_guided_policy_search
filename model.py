
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
        self.name = name
        self.ix = 5
        self.iu = 2
        self.delT = 0.1
        
    def forwardDyn(self,x,u,idx=None):

                # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
     
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        v_p = x[:,3]
        w_p = x[:,4]

        delv = u[:,0]
        delw = u[:,1]
        
        v = v_p + delv
        w = w_p + delw
        
        # output
        f = np.zeros_like(x)
        f[:,0] = v * np.cos(x3)
        f[:,1] = v * np.sin(x3)
        f[:,2] = w
        f[:,3] = 1 / self.delT * delv
        f[:,4] = 1 / self.delT * delw
        
        return np.squeeze(x + f * self.delT)
    
    def diffDyn(self,x,u):

        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        v_p = x[:,3]
        w_p = x[:,4]
        
        delv = u[:,0]
        delw = u[:,1]    
        
        v = v_p + delv
        w = w_p + delw
        
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
        fx[:,0,3] = self.delT * np.cos(x3)
        fx[:,0,4] = 0.0
        fx[:,1,3] = self.delT * np.sin(x3)
        fx[:,1,4] = 0.0
        fx[:,2,3] = 0.0
        fx[:,2,4] = self.delT
        fx[:,3,3] = 1.0
        fx[:,4,4] = 1.0
          
        fu = np.zeros((N,self.ix,self.iu))
        fu[:,0,0] = self.delT * np.cos(x3)
        fu[:,0,1] = 0.0
        fu[:,1,0] = self.delT * np.sin(x3)
        fu[:,1,1] = 0.0
        fu[:,2,0] = 0.0
        fu[:,2,1] = self.delT
        fu[:,3,0] = 1.0
        fu[:,4,1] = 1.0
           
        
        return np.squeeze(fx) , np.squeeze(fu)

class generalSlip :
    def __init__(self,name,alpha):
        self.name = name
        self.ix = 5
        self.iu = 2
        self.delT = 0.1
        self.alpha = alpha
        
    def forwardDyn(self,x,u,idx=None):

        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
     
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        v_p = x[:,3]
        w_p = x[:,4]

        delv = u[:,0]
        delw = u[:,1]
        
        v = v_p + delv
        w = w_p + delw

        # velocity
        vx = v + self.alpha[0] * v + self.alpha[1] * np.abs(w) + self.alpha[2] * v * np.abs(w)
        vy = 0 + self.alpha[3] * v + self.alpha[4] * w + self.alpha[5] * v * w
        vthe = w + self.alpha[6] * v + self.alpha[7] * w + self.alpha[8] * v * w

        # output
        f = np.zeros_like(x)
        f[:,0] = vx * np.cos(x3) - vy * np.sin(x3)
        f[:,1] = vx * np.sin(x3) + vy * np.cos(x3)
        f[:,2] = vthe
        f[:,3] = 1 / self.delT * delv
        f[:,4] = 1 / self.delT * delw
        
        return np.squeeze(x + f * self.delT)
    
    def diffDyn(self,x,u):

        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        v_p = x[:,3]
        w_p = x[:,4]
        
        delv = u[:,0]
        delw = u[:,1]    
        
        v = v_p + delv
        w = w_p + delw

        # velocity
        vx = v + self.alpha[0] * v + self.alpha[1] * np.abs(w) + self.alpha[2] * v * np.abs(w)
        vy = 0 + self.alpha[3] * v + self.alpha[4] * w + self.alpha[5] * v * w
        vthe = w + self.alpha[6] * v + self.alpha[7] * w + self.alpha[8] * v * w

        # velocity_diff
        vx_x = 1 + self.alpha[0] + self.alpha[2] * np.abs(w)
        vx_t = ( self.alpha[1] + self.alpha[2] * v ) * ( (w >= 0) * 1 - (w < 0) * 1 )
        vy_x = self.alpha[3] + self.alpha[5] * w
        vy_t = self.alpha[4] + self.alpha[5] * v
        vt_x = self.alpha[6] + self.alpha[8] * w
        vt_t = 1 + self.alpha[7] + self.alpha[8] * v
        
        fx = np.zeros((N,self.ix,self.ix))
        fx[:,0,0] = 1.0
        fx[:,0,1] = 0.0
        fx[:,0,2] = self.delT * ( - vx * np.sin(x3) - vy * np.cos(x3) ) 
        fx[:,1,0] = 0.0
        fx[:,1,1] = 1.0
        fx[:,1,2] = self.delT * ( vx * np.cos(x3) - vy * np.sin(x3) )
        fx[:,2,0] = 0.0
        fx[:,2,1] = 0.0
        fx[:,2,2] = 1.0

        fx[:,0,3] = self.delT * ( ( vx_x ) * np.cos(x3) \
                                - ( vy_x ) * np.sin(x3) )
        fx[:,0,4] = self.delT * ( ( vx_t ) * np.cos(x3) \
                                - ( vy_t ) * np.sin(x3) )

        fx[:,1,3] = self.delT * ( (vx_x) * np.sin(x3) \
                                + (vy_x) * np.cos(x3) )
        fx[:,1,4] = self.delT * ( (vx_t) * np.sin(x3) \
                                + (vy_t) * np.cos(x3) )
        
        fx[:,2,3] = self.delT * vt_x
        fx[:,2,4] = self.delT * vt_t
        
        fx[:,3,3] = 1.0
        fx[:,4,4] = 1.0
          
        fu = np.zeros((N,self.ix,self.iu))
        fu[:,0,0] = self.delT * ( ( vx_x ) * np.cos(x3) \
                                - ( vy_x ) * np.sin(x3) )
        fu[:,0,1] = self.delT * ( ( vx_t ) * np.cos(x3) \
                                - ( vy_t ) * np.sin(x3) )

        fu[:,1,0] = self.delT * ( (vx_x) * np.sin(x3) \
                                + (vy_x) * np.cos(x3) )
        fu[:,1,1] = self.delT * ( (vx_t) * np.sin(x3) \
                                + (vy_t) * np.cos(x3) )

        fu[:,2,0] = self.delT * vt_x
        fu[:,2,1] = self.delT * vt_t

        fu[:,3,0] = 1.0
        fu[:,4,1] = 1.0
        
        return np.squeeze(fx) , np.squeeze(fu)

    def alpha_update(self,x_fit,u_fit,num_fit,N,num_ini) :
        
        b = np.zeros((N,3,1,num_fit,num_ini))
        C = np.zeros((N,3,9,num_fit,num_ini))

        b_mat = np.zeros((N*3*num_fit*num_ini,1))
        C_mat = np.zeros((N*3*num_fit*num_ini,9))

        index = -1
        for j in range(num_ini) :
            for i in range(num_fit) :
                for ip in range(N) : 
                        index += 1

                        # position
                        x_current = x_fit[j][ip,0:3,i]
                        x_next = x_fit[j][ip+1,0:3,i]

                        # robot input
                        v = x_fit[j][ip,3,i] + u_fit[j][ip,0,i]
                        w = x_fit[j][ip,4,i] + u_fit[j][ip,1,i]

                        # rotation matrix
                        R = np.zeros((3,3))
                        R[0,0] = np.cos(x_current[2])
                        R[0,1] = - np.sin(x_current[2])
                        R[1,0] = np.sin(x_current[2])
                        R[1,1] = np.cos(x_current[2])
                        R[2,2] = 1

                        # V IDD
                        v_idd = np.zeros(3)
                        v_idd[0] = v
                        v_idd[2] = w

                        b[ip,:,:,i,j] = np.matmul(np.linalg.inv(R),np.expand_dims(x_next-x_current,axis=1)) / self.delT - np.expand_dims(v_idd,axis=1)
                        C[ip,:,:,i,j] = np.zeros((3,9))
                        C[ip,0,0,i,j] = v
                        C[ip,0,1,i,j] = np.abs(w)
                        C[ip,0,2,i,j] = v * np.abs(w)

                        C[ip,1,3,i,j] = v
                        C[ip,1,4,i,j] = w
                        C[ip,1,5,i,j] = v * w

                        C[ip,2,6,i,j] = v
                        C[ip,2,7,i,j] = w
                        C[ip,2,8,i,j] = v * w

                        C_mat[3*index:3*(index+1),:] = C[ip,:,:,i,j]
                        b_mat[3*index:3*(index+1),:] = b[ip,:,:,i,j]
                        # if ip < 10 :
                        #     print np.matmul(C[ip,:,:,i,j],np.expand_dims(self.alpha,axis=1)) - b[ip,:,:,i,j]

        alpha = np.matmul(np.linalg.pinv(C_mat), b_mat)
        # print self.alpha
        print np.squeeze(alpha)
        self.alpha = np.squeeze(alpha)

        return None


    
    
    
    