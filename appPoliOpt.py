
# coding: utf-8

# In[9]:

from __future__ import division
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
import tensorflow as tf
import GPy
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
class appPoliOpt:
    def __init__(self,name,hidden_num,maxIter,stepSize,ix,iu,N,flagNN):
        # name
        self.name = name
        self.maxIter = maxIter
        self.stepSize = stepSize
        self.hidden_num = hidden_num

        # size
        self.ix = ix
        self.iu = iu
        self.N = N

        # flag NN or GP?
        self.flagNN = flagNN
        
        if self.flagNN == True :
        # Initialize tensorflow setting
            self.setInitial(self.maxIter,self.stepSize)
        else :
            self.kern_var = 1.0
            self.lengthscale = 1.0
            self.kernel = GPy.kern.RBF(input_dim=self.ix, variance=self.kern_var \
                            ,lengthscale=self.lengthscale)
            
            pass
        
    def setInitial(self,maxIter,stepSize) :
        
        self.stepSize = stepSize
        self.maxIter = maxIter
        
        # tensorflow variables & parameters
        self.input_x = tf.placeholder(tf.float32, [None,self.ix])
        self.input_y = tf.placeholder(tf.float32, [None,self.iu])
        
        self.layer1 = fully_connected(self.input_x, self.hidden_num, scope= "layer1_app")
        # self.layer2 = fully_connected(self.layer1, self.hidden_num, scope = "layer2_app")
        self.Y_pred = fully_connected(self.layer1, self.iu, scope = "layer3_app", activation_fn = None)
        # warning!! : this code implicates that the size of input_y(iu) is fixed as 2
        self.fx1 = tf.gradients(self.Y_pred[:,0],self.input_x)
        self.fx2 = tf.gradients(self.Y_pred[:,1],self.input_x)

         # loss function
        self.u_error = self.Y_pred - self.input_y
        self.loss = tf.reduce_mean(tf.square(self.u_error))
        
        self.optimizer = tf.train.GradientDescentOptimizer(self.stepSize).minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)   
        
    def setEnv(self,maxIter,stepSize) :
        
        if self.flagNN == True :
            self.stepSize = stepSize
            self.maxIter = maxIter
            # self.setInitial(maxIter,stepSize)
            self.optimizer = tf.train.GradientDescentOptimizer(self.stepSize).minimize(self.loss)
            # self.init = tf.global_variables_initializer()
            # self.sess = tf.Session()
            # self.sess.run(self.init)
        else :
            # self.kernel = GPy.kern.RBF(input_dim=self.ix, variance=self.kern_var,lengthscale=self.lengthscale)
            pass
    def setData(self,x,u) :
        
        # change data
        self.state_label = x
        self.input_label = u
        
    def meanOpt(self,set_num) :
        
        if self.flagNN == True :
            batch_size = 32
            num_batch = int((set_num*self.N) / batch_size)
            loss = np.zeros(num_batch)
            x_batch = np.zeros((batch_size,self.ix))
            y_batch = np.zeros((batch_size,self.iu))
            for i in range(self.maxIter) :
                for j in range(num_batch) :
                    # index = np.random.randint(set_num*self.N, size=batch_size)
                    index = np.random.choice(set_num*self.N,size=batch_size,replace=False)
                    x_batch = self.state_label[index,:]
                    y_batch = self.input_label[index,:]
                    self.sess.run(self.optimizer, feed_dict={self.input_x: x_batch, self.input_y: y_batch})
                    loss[j] = self.sess.run(self.loss,feed_dict={self.input_x: x_batch, self.input_y: y_batch})
                if i % 40 == 0 :
                    print "epoch " + str(i+1) + ", Training Loss= " + "{:.6f}".format(np.mean(loss))
            print("Optimization - approximated NN policy is optimized")
        else :
            self.m = GPy.models.GPRegression(self.state_label,self.input_label,self.kernel)
            self.m.optimize(messages=True)
            self.lengthscale = self.m.kern.lengthscale
            self.kern_var = self.m.kern.variance
            print("Optimization - approximated GP policy is optimized")

        
    def getPolicy(self,x) :
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            x = np.expand_dims(x,axis=0)
        
        # stochastic policy
        
        # deterministic policy
        if self.flagNN == True :
            return self.sess.run(self.Y_pred,feed_dict={self.input_x: x})
        else :
            return self.m.predict(x)[0]

    def getMean(self,x) :

        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            x = np.expand_dims(x,axis=0)

        u = self.getPolicy(x)

        return np.hstack((x,u))

            
    
            
    def update(self,x,u,set_num) :
        
        # data setting
        self.setData(x,u)
        
        # mean optimization - backpropagation
        self.meanOpt(set_num)

    def jacobian(self,x) :

        ix = self.ix
        iu = self.iu
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
        else :
            N = np.size(x,axis = 0)

        # if self.flagNN == True :
        if False == True :
        
            fx1_val = np.transpose(np.array( self.sess.run(self.fx1,feed_dict={self.input_x: x}) ),axes=[1,0,2])
            fx2_val = np.transpose(np.array( self.sess.run(self.fx2,feed_dict={self.input_x: x}) ),axes=[1,0,2])
            fx = np.hstack((fx1_val,fx2_val))

        else :

            h = pow(2,-17)
            eps_x = np.identity(ix)
            x_mat = np.expand_dims(x,axis=2)
            x_diag = np.tile(x_mat,(1,1,ix))

            x_aug = x_diag + eps_x * h
            x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(ix),ix))
            # print_np(x_aug)
            y_nominal = self.getPolicy(x)
            y_change = self.getPolicy(x_aug)
            y_change = np.reshape(y_change,(N,ix,iu))
            fx = (y_change - np.reshape(y_nominal,(N,1,iu)) ) / h
            fx = np.transpose(np.reshape(fx,(N,ix,iu)),[0,2,1])
        return fx




# In[183]:

# # x,u, Quu_save, Quu_inv_save = i1.update(x0,u0)
# xData = x[0:400,:]
# myPolicy = poliOpt('test',xData,u,Quu,10,5000,0.01)
# W1,b1,W2,b2,var = myPolicy.update(xData,u,Quu)

