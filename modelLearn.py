
# coding: utf-8

# In[2]:

from __future__ import division
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
import tensorflow as tf
import GPflow
from termcolor import colored
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))


# In[11]:

class localModelLearn :
    def __init__(self,name,ix,iu,N,movHorizon,xuxDataIni,bool_GP = False):
        self.name = name
        self.N = N
        self.ix = ix
        self.iu = iu
        self.iData = self.ix * 2 + self.iu
        
        self.A = np.zeros((N,ix,ix))
        self.B = np.zeros((N,ix,iu))
        self.C = np.zeros((N,ix))  
        self.F = np.zeros((N,ix,ix))
        
        self.xuxData = xuxDataIni
        self.dataNum = 0
          
        self.movHorizon = movHorizon
        
        self.bool_GP = bool_GP
        
        # priori 
        self.mean_bar = np.zeros((N,self.iData))
        self.cov_bar = np.zeros((N,self.iData,self.iData))
#         for i in range(self.N) : 
#             self.cov_bar[i,:,:] = np.cov(self.xuxData[i,:,:])
#         print(self.cov_bar)
 
        self.m = 0
        self.n0 = 0
        
        self.Phi = self.n0 * self.cov_bar
        self.mu0 = self.mean_bar
        
        # posteriori
        self.mu = np.zeros((self.N,self.iData))
        self.cov = np.zeros((self.N,self.iData,self.iData))
                
    def data_aug(self,x_fit,u_fit_m,bool_first = False) :
        
        N = self.N
        iData = self.iData
        
        xuxDataNew = np.hstack((x_fit[0:N,:,:],u_fit_m,x_fit[1:N+1,:,:]))
        
        if bool_first == True : 
            self.xuxData = xuxDataNew
            self.m = 1
            self.n0 = 1
        else : 
            self.xuxData = np.dstack((self.xuxData,xuxDataNew))
        
        self.dataNum = self.xuxData.shape[2]
        if self.dataNum > self.movHorizon :
            self.xuxData = self.xuxData[:,:,self.dataNum-self.movHorizon:self.dataNum]  
            self.dataNum = self.movHorizon
            
    def prior_update(self,mean_hat) :
        
        N = self.N
        iData = self.iData
        ix = self.ix
        iu = self.iu
        
        if not self.dataNum == 0 :
            if self.bool_GP == False :
                # priori 
                self.mean_bar = np.mean(self.xuxData, axis=2)
                self.cov_bar = np.zeros((N,iData,iData))
                for i in range(self.N) : 
                    self.cov_bar[i,:,:] = np.cov(self.xuxData[i,:,:])
            else :
                self.mean_bar = np.zeros((N,iData))
                self.cov_bar = np.zeros((N,iData,iData))
                
                k = GPflow.kernels.RBF(5, lengthscales=0.3)
                input_data = self.xuxData[:,0:ix+iu,:]
                input_data = np.reshape(np.transpose(input_data,[0,2,1]),[N*self.dataNum,ix+iu])
                output_data = self.xuxData[:,ix+iu:iData,:]
                output_data = np.reshape(np.transpose(output_data,[0,2,1]),[N*self.dataNum,ix])
                m = GPflow.gpr.GPR(input_data, output_data, kern=k)
                m.likelihood.variance = 0.01
                m.optimize()
                mean,var =  np.squeeze( m.predict_y(mean_hat[:,0:ix+iu] ) )
                self.mean_bar = np.hstack( (mean_hat[:,0:ix+iu], mean ) )
                for i in range(self.N) :
  
                    self.cov_bar[i,:,:] = np.cov(self.xuxData[i,:,:])
                
            # update prior parameters    
            self.Phi = self.n0 * self.cov_bar
            self.mu0 = self.mean_bar
              
    def poste_update(self,mean_hat,cov_hat,num_hat) : 
                
        # posteriori estimation
        temp = np.squeeze( np.matmul(np.expand_dims(mean_hat - self.mu0,axis=2),np.transpose(np.expand_dims(                 mean_hat - self.mu0, axis=2),[0,2,1]) ) )
        self.mu = ( self.m * self.mu0 + num_hat * mean_hat ) / ( self.m + num_hat)
        self.cov = 1 / (num_hat + self.n0) * (self.Phi + num_hat * cov_hat                                               + num_hat * self.m / (num_hat + self.m) * temp )
        
        
    def set_linear(self) :
        
        N = self.N
        ix = self.ix
        iu = self.iu
        iData = self.iData
        
        # This code should be changed as matrix calculation
        for i in range(N) : 
            temp = np.dot( np.linalg.inv( self.cov[i,0:self.ix+self.iu,0:self.ix+self.iu] ), self.cov[i,0:self.ix+self.iu,self.ix+self.iu:self.iData] )
            self.A[i,:,:] = temp[0:ix,:].T
            self.B[i,:,:] = temp[ix:ix+iu,:].T
            self.C[i,:] = self.mu[i,self.ix+self.iu:self.iData] - np.dot(temp.T, self.mu[i,0:self.ix+self.iu] )
            self.F[i,:,:] = self.cov[i,self.ix+self.iu:self.iData,self.ix+self.iu:self.iData] -                             np.dot( np.dot(temp.T, self.cov[i,0:self.ix+self.iu,0:self.ix+self.iu] ), temp )
                
    def update(self,x_fit,u_fit_m) :
        
        N = self.N
        iData = self.iData
        num_hat = x_fit.shape[2]
        
        # empirical mean and covariance from current iteration
        xuxDataNew = np.hstack((x_fit[0:N,:,:],u_fit_m,x_fit[1:N+1,:,:]))
        mean_hat = np.mean(xuxDataNew,axis=2)
        cov_hat = np.zeros((N,iData,iData))
        
        for i in range(self.N) : 
            cov_hat[i,:,:] = np.cov(xuxDataNew[i,:,:])
            
        
        self.prior_update(mean_hat)
        self.poste_update(mean_hat,cov_hat,num_hat)
        self.set_linear()
        
    
    def diffDyn(self,x,u) :
           
        return self.A, self.B 
    
    def forwardDyn(self,x,u,idx):
        
        return np.squeeze( np.dot(self.A[idx,:,:],x) + np.dot(self.B[idx,:,:],u) + self.C[idx,:] )
    
    
class localPolicyLearn :
    def __init__(self,name,ix,iu,N,movHorizon,xuDataIni):
        self.name = name
        self.N = N
        self.ix = ix
        self.iu = iu
        self.iData = self.ix + self.iu
        
        self.K = np.zeros((N,iu,ix))
        self.k = np.zeros((N,iu))
        
        self.xuData = xuDataIni
        self.dataNum = 0
          
        self.movHorizon = movHorizon
        
        # priori 
        self.mean_bar = np.zeros((N,self.iData))
        self.cov_bar = np.zeros((N,self.iData,self.iData))
 
        self.m = 0
        self.n0 = 0
        
        self.Phi = self.n0 * self.cov_bar
        self.mu0 = self.mean_bar
        
        # posteriori
        self.mu = np.zeros((self.N,self.iData))
        self.cov = np.zeros((self.N,self.iData,self.iData))
                
    def data_aug(self,x_fit,u_fit_p,bool_first = False) :
        
        N = self.N
        iData = self.iData
        
        xuDataNew = np.hstack((x_fit[0:N,:,:],u_fit_p))
        
        if bool_first == True : 
            self.xuData = xuDataNew
            self.m = 1
            self.n0 = 1
        else : 
            self.xuData = np.dstack((self.xuData,xuDataNew))
        
        self.dataNum = self.xuData.shape[2]
        if self.dataNum > self.movHorizon :
            self.xuData = self.xuData[:,:,self.dataNum-self.movHorizon:self.dataNum]  
            self.dataNum = self.movHorizon
            
    def prior_update(self) :
        
        N = self.N
        iData = self.iData
        
        if not self.dataNum == 0 : 
            # priori 
            self.mean_bar = np.mean(self.xuData, axis=2)
            self.cov_bar = np.zeros((N,iData,iData))
            for i in range(self.N) : 
                self.cov_bar[i,:,:] = np.cov(self.xuData[i,:,:])

            # update prior parameters    
            self.Phi = self.n0 * self.cov_bar
            self.mu0 = self.mean_bar
              
    def poste_update(self,mean_hat,cov_hat,num_hat) : 
                
        # posteriori estimation
        temp = np.squeeze( np.matmul(np.expand_dims(mean_hat - self.mu0,axis=2),np.transpose(np.expand_dims(                 mean_hat - self.mu0, axis=2),[0,2,1]) ) )
        self.mu = ( self.m * self.mu0 + num_hat * mean_hat ) / ( self.m + num_hat)
        self.cov = 1 / (num_hat + self.n0) * (self.Phi + num_hat * cov_hat                                               + num_hat * self.m / (num_hat + self.m) * temp )
        
        
    def set_linear(self) :
        
        N = self.N
        ix = self.ix
        iu = self.iu
        iData = self.iData
        
        # This code should be changed as matrix calculation
        for i in range(N) : 
            temp = np.dot( np.linalg.inv( self.cov[i,0:self.ix,0:self.ix] ), self.cov[i,0:self.ix,self.ix:self.iData] )
            self.K[i,:,:] = temp.T
            self.k[i,:] = self.mu[i,self.ix:self.iData] - np.dot(temp.T, self.mu[i,0:self.ix] )
            
                
    def update(self,x_fit,u_fit_p) :
        
        N = self.N
        iData = self.iData
        num_hat = x_fit.shape[2]
        
        # empirical mean and covariance from current iteration
        xuDataNew = np.hstack((x_fit[0:N,:,:],u_fit_p))
        mean_hat = np.mean(xuDataNew,axis=2)
        cov_hat = np.zeros((N,iData,iData))
        
        for i in range(self.N) : 
            cov_hat[i,:,:] = np.cov(xuDataNew[i,:,:])
            
        
        self.prior_update()
        self.poste_update(mean_hat,cov_hat,num_hat)
        self.set_linear()
        
    
    def diffDyn(self,x,u) :
           
        return self.A, self.B 
    
    def forwardDyn(self,x,u,idx):
        
        return np.squeeze( np.dot(self.A[idx,:,:],x) + np.dot(self.B[idx,:,:],u) + self.C[idx,:] )
    
    
        
        


# In[ ]:



