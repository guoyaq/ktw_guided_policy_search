
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


# In[3]:

# i1 = iLQR.iLQR('doublePendulum',200)
# x0 = np.zeros(4)
# u0 = np.ones( (200,2) )
# x0[0] = -np.pi/2
# x0[1] = 0.
# x,u, Quu_save, Quu_inv_save = i1.update(x0,u0)


# In[4]:

class poliOpt:
    def __init__(self,name,ix,iu,hidden_num,maxIter,stepSize):
        # name
        self.name = name
        self.maxIter = maxIter
        self.stepSize = stepSize
        self.hidden_num = hidden_num
        # data
        # self.state_label = x
        # self.input_label = u
        # self.var_inv_label = Q
        # self.dual = dual
        
        # size
        self.ix = ix
        self.iu = iu
        
        # policy variance <- constant when optimization was done
        self.policy_var = np.zeros((self.iu,self.iu))
        
        # Initialize tensorflow setting
        self.setInitial(self.stepSize,self.maxIter)
        
    def setInitial(self,stepSize,maxIter) :
        self.stepSizz = stepSize
        self.maxIter = maxIter
        
        # tensorflow variables & parameters
        self.Weight = tf.placeholder(tf.float32, [None,self.iu,self.iu])
        self.input_x = tf.placeholder(tf.float32, [None,self.ix])
        self.input_y = tf.placeholder(tf.float32, [None,self.iu])
        self.dual_tf = tf.placeholder(tf.float32,[None,self.iu])
        
        # variables
        self.W1 = tf.Variable(tf.random_normal(shape = [self.ix,self.hidden_num]),dtype = tf.float32)
        self.b1 = tf.Variable(tf.random_normal(shape = [self.hidden_num]),dtype =  tf.float32)

        self.W2 = tf.Variable(tf.random_normal(shape = [self.hidden_num,self.iu]),dtype =  tf.float32)
        self.b2 = tf.Variable(tf.random_normal(shape = [self.iu]),dtype = tf.float32)

        self.layer1 = tf.nn.softplus(tf.add(tf.matmul(self.input_x,self.W1),self.b1))
        self.layer2 = tf.add(tf.matmul(self.layer1, self.W2), self.b2)
        self.Y_pred = self.layer2
        
         # loss function
        self.u_error = tf.expand_dims(self.Y_pred - self.input_y,2)
        self.loss = tf.reduce_mean(tf.matmul( tf.matmul(tf.transpose(self.u_error,perm=[0,2,1]), self.Weight
                    ),self.u_error) ) + 2 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.Y_pred,self.dual_tf),1))    
        
        self.optimizer = tf.train.AdamOptimizer(self.stepSize).minimize(self.loss)
        # self.optimizer = tf.train.GradientDescentOptimizer(self.stepSize).minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        
    def setData(self,x,u,Q,dual):
        
        # change data
        self.state_label = x
        self.input_label = u
        self.var_inv_label = Q
        self.dual = dual
                
    def meanOpt(self) :

        for i in range(self.maxIter) :
            self.sess.run(self.optimizer, feed_dict={self.input_x: self.state_label,                                                                       self.input_y: self.input_label,self.Weight : self.var_inv_label,
                          self.dual_tf : self.dual})
            if i % 500 == 0 :
                loss_temp = self.sess.run(self.loss,feed_dict={self.input_x: self.state_label,                                                                             self.input_y: self.input_label,self.Weight : self.var_inv_label,                                                                 self.dual_tf : self.dual})
                print(loss_temp)
            # if loss_temp < 1 :
            #    break
        
        print("Optimization - policy mean is done")
        
    def varOpt(self) :
        
        # data
        Q = self.var_inv_label
        
        # optimization
        cov_inv = Q.sum(axis=0) / np.size(Q,axis=0)
        self.policy_var = np.linalg.inv(cov_inv)
        
        print("Optimization - policy variance is done")

    def getWeight(self) :
        
        return self.sess.run([self.W1, self.b1, self.W2, self.b2])
        
    def getPolicy(self,x) :
        
        # stochastic policy
        
        # deterministic policy
        return self.sess.run(self.Y_pred,feed_dict={self.input_x: x})
    
    
    def gaussPDF(self,x,u) :
        
        ndim = np.ndim(u)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # mean & variance of policy
        mean = self.sess.run(self.Y_pred,feed_dict={self.input_x: x})
        var = self.policy_var
        
        # expand dims to tensor
        u = np.expand_dims(u,axis=2)
        mean = np.expand_dims(mean,axis=2)

        error = u - mean
        var_inv = np.tile( np.linalg.inv(var), (N,1,1) )
        
        temp = np.matmul(np.matmul( np.transpose(error,(0,2,1)),var_inv),error)
        # print_np(var)
        p = 1 / ( np.sqrt( np.abs( 2 * np.pi * np.linalg.det(var) ))) * np.exp( - 0.5 * temp)
            
        return np.squeeze(p)
            
    def update(self,x,u,Q,dual) :
        
        # data setting
        self.setData(x,u,Q,dual)
        
        # mean optimization - backpropagation
        self.meanOpt()
        
        # variance optimization
        self.varOpt()

        # get parameters of network
        W1,b1,W2,b2 = self.getWeight()
        
        # return parameter, variance of policy
        return W1,b1,W2,b2, self.policy_var
        
        
    

        
        


# In[7]:

# # x,u, Quu_save, Quu_inv_save = i1.update(x0,u0)
# xData = x[0:200,:]
# dual = np.ones((200,2)) * 0.1
# myPolicy = poliOpt('test',xData,u,Quu_save,dual,10,1000)
# W1,b1,W2,b2,var = myPolicy.update(xData,u,Quu_save,dual)


# In[8]:

# x1 = np.ones((100,4))
# u1 = np.ones((100,2))
# p = myPolicy.gaussPDF(x1,u1)
# print_np(p)


# In[ ]:



