
# coding: utf-8

# In[9]:

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
from tensorflow.contrib.layers import fully_connected


# In[10]:

import model
import cost
import iLQR


# In[11]:

# # Initial trajectory distribution from iLQR with low number of iterations
# i1 = iLQR.iLQR('unicycle',400,100)
# x0 = np.zeros(3)
# x0[0] = 5
# x0[1] = 5
# x0[2] = np.pi / 2
# u0 = np.zeros( (400,2) )
# x,u, Quu, Quu_inv = i1.update(x0,u0)


# In[182]:

class poliOpt:
    def __init__(self,name,hidden_num,maxIter,stepSize,ix,iu,N):
        # name
        self.name = name
        self.maxIter = maxIter
        self.stepSize = stepSize
        self.hidden_num = hidden_num
        
        # data
        # self.state_label = x
        # self.input_label = u
        # self.var_inv_label = Q
        
        # size
        self.ix = ix
        self.iu = iu
        self.N = N
        
        # policy variance <- constant when optimization was done
        self.policy_var = np.zeros((self.N,self.iu,self.iu))
        self.policy_var_inv = np.zeros((self.N,self.iu,self.iu))
        
        # for RNN
        self.n_steps = 2
        
        # Initialize tensorflow setting
        self.setInitial(self.maxIter,self.stepSize)


    def setInitial(self,maxIter,stepSize) :
        
        self.stepSize = stepSize
        self.maxIter = maxIter
        
        # tensorflow variables & parameters
        self.Weight = tf.placeholder(tf.float32, [None,self.iu,self.iu])
        self.input_x = tf.placeholder(tf.float32, [None,self.ix])
        self.input_aug = tf.placeholder(tf.float32, [None,self.n_steps,self.ix])
        self.input_y = tf.placeholder(tf.float32, [None,self.iu])
        
        # variables
        basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_num)
        outputs,states = tf.nn.dynamic_rnn(basic_cell, self.input_aug, dtype=tf.float32)
        self.layer1 = states.h
        # self.layer1 = fully_connected(tf.reshape(self.input_aug,[-1,6]), self.hidden_num, scope= "layer1")
        self.layer2 = fully_connected(self.layer1, self.hidden_num, scope = "layer2")
        self.Y_pred = fully_connected(self.layer2, self.iu, scope = "layer3", activation_fn = None)
        
         # loss function
        self.u_error = tf.expand_dims(self.Y_pred - self.input_y,2)
        self.loss = tf.reduce_mean(tf.matmul( tf.matmul(tf.transpose(self.u_error,perm=[0,2,1]), self.Weight
                    ),self.u_error) )
        
        # self.optimizer = tf.train.AdamOptimizer(self.stepSize).minimize(self.loss)
        self.optimizer = tf.train.GradientDescentOptimizer(self.stepSize).minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)   
        
    def setEnv(self,maxIter,stepSize) :
        
        self.stepSize = stepSize
        self.maxIter = maxIter
        self.optimizer = tf.train.GradientDescentOptimizer(self.stepSize).minimize(self.loss)
        
    def setData(self,x,u,Q,Q_set) :
        
        # change data
        self.state_label = x
        self.input_label = u
        self.var_inv_label = Q
        self.var_inv_set_label = Q_set
        
    def meanOpt(self,set_num) :
        
        
        
        for i in range(self.maxIter) :
            index = np.random.randint(0,set_num)
            # x_batch = self.state_label[index*self.N:(index+1)*self.N,:]
            u_batch = self.input_label[index*self.N:(index+1)*self.N,:]
            W_batch = self.var_inv_label[index*self.N:(index+1)*self.N,:,:]
            x1_temp = self.state_label[index*self.N:(index+1)*self.N,:]
            x2_temp = x1_temp
            x2_temp[1:,:] = x1_temp[0:-1,:]
            x_batch = np.dstack((x2_temp,x1_temp))
            x_batch = np.transpose(x_batch,[0,2,1])
            
            self.sess.run(self.optimizer, feed_dict={self.input_aug: x_batch, self.input_y: u_batch, self.Weight : W_batch})
            if i % 2000 == 0 :
                loss_temp = self.sess.run(self.loss,feed_dict={self.input_aug: x_batch, self.input_y: u_batch, self.Weight : W_batch})
                print(loss_temp)
                if loss_temp < 1e-6 :
                    break
        print("Optimization - policy mean is done")
        
    def varOpt(self) :
        
        # data
        Q_set = self.var_inv_set_label
        
        # optimization
        for i in range(self.N) :
            cov_inv = Q_set[i,:,:,:].sum(axis=2) / np.size(Q_set,axis=3)
            self.policy_var_inv[i,:,:] = cov_inv
            self.policy_var[i,:,:] = np.linalg.inv(cov_inv)
        
        print("Optimization - policy variance is done")

    def getWeight(self) :
        
        with tf.variable_scope('layer1', reuse=True):
            self.W1 = tf.get_variable('weights')
            self.b1 = tf.get_variable('biases')
            
        with tf.variable_scope('layer2', reuse=True):
            self.W2 = tf.get_variable('weights')
            self.b2 = tf.get_variable('biases')
            
        with tf.variable_scope('layer3', reuse=True):
            self.W3 = tf.get_variable('weights')
            self.b3 = tf.get_variable('biases')
            
        return self.sess.run([self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
        
    def getPolicy(self,x) :
        
        ndim = np.ndim(x)
        if ndim == 2: # 1 step state & input
            x = np.expand_dims(x,axis=0)
        
        # stochastic policy
        
        # deterministic policy
        return self.sess.run(self.Y_pred,feed_dict={self.input_aug: x}), self.policy_var
    
            
    def update(self,x,u,Q,Q_set,set_num) :
        
        # data setting
        self.setData(x,u,Q,Q_set)
        
        # mean optimization - backpropagation
        self.meanOpt(set_num)
        
        # variance optimization
        self.varOpt()

        # get parameters of network
        # W1,b1,W2,b2,W3,b3 = self.getWeight()
        
        # return parameter, variance of policy
        return self.policy_var, self.policy_var_inv
    
    def jacobian(self,x) :
        # This require a lot of computational time.. so it should be more fancy
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
        else :
            N = np.size(x,axis = 0)
            
        # temp = [[] for i in range(self.iu)]
        # grad_nn = [[] for i in range(self.iu)]
        # out = [[] for i in range(self.iu)]
        # for i in range(self.iu) :
        #     temp[i] = self.Y_pred[:,i]
        #     grad_nn[i] = tf.gradients(temp[i],self.input_x)
        #     out[i] = np.transpose( np.array( self.sess.run(grad_nn[i],feed_dict={self.input_x: x}) ) , (1,0,2))
        # return np.hstack((out[i] for i in range(self.iu)))
        if self.name == 'unicylce' :
            temp1 = self.Y_pred[:,0]
            temp2 = self.Y_pred[:,0]
            grad_nn1 = tf.gradients(temp1,self.input_x)
            grad_nn2 = tf.gradients(temp2,self.input_x)
            out1 = np.transpose( np.array( self.sess.run(grad_nn1,feed_dict={self.input_x: x}) ) , (1,0,2))
            out2 = np.transpose( np.array( self.sess.run(grad_nn2,feed_dict={self.input_x: x}) ) , (1,0,2))
            output = np.hstack((out1,out2))
            
        return output
        


# In[183]:

# # x,u, Quu_save, Quu_inv_save = i1.update(x0,u0)
# xData = x[0:400,:]
# myPolicy = poliOpt('test',xData,u,Quu,10,5000,0.01)
# W1,b1,W2,b2,var = myPolicy.update(xData,u,Quu)

