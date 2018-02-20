# coding: utf-8
from __future__ import division
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
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

class trajOpt:
    def __init__(self,name,horizon,Model,Cost,maxIter):
        # class name
        self.name = name

        # model & cost
        self.model = Model
        self.cost = Cost
           
        # input & output size
        self.ix = self.model.ix
        self.iu = self.model.iu

        # iLQR parameters
        self.verbosity = True
        self.dlamda = 1
        self.lamda = 1e-7
        self.lamdaFactor = 1.6
        self.lamdaMax = 1e-6
        self.lamdaMin = 1e-8
        self.tolFun = 1e-2
        self.tolGrad = 1e-4
        self.maxIter = maxIter
        self.zMin = 0
        self.last_head = True
        self.dV = np.zeros((1,2))
        self.N = horizon
        
        # dual variable
        self.eta = 1
        
        # KL constraint variable
        self.epsilon = 2

        # state & input contstraints variables
        self.phi = np.ones(self.N+1) * 1.0
        self.tolConst = 0.1

        # flag const
        self.flag_const = self.cost.flag_const
        self.flagAppNN = False

        

        # initial storage
        self.initStorage()

    def initStorage(self) :
        
        # initial trajectory
        self.x0 = np.zeros(self.model.ix)
        self.x0cov = np.identity(self.model.ix)
        self.x = np.zeros((self.N+1,self.model.ix))
        self.S = np.tile(0.01*np.identity(self.model.ix),[self.N+1,1,1])
        self.u = np.zeros((self.N,self.model.iu))
        self.C = np.tile(0.01*np.identity(self.model.ix+self.model.iu),[self.N+1,1,1])

        # variables for constraints # self.cost.ic,
        mu_ini = 1e-2 * self.flag_const
        self.Mu_e = np.tile(np.identity(self.cost.ic),(self.N+1,1,1)) * mu_ini * self.flag_const
        self.lam = np.tile(np.identity(self.cost.ic),(self.N+1,1,1)) * 0.01 * self.flag_const

        
        # next trajectroy
        self.xnew = np.zeros((self.N+1,self.model.ix))
        self.unew = np.zeros((self.N,self.model.iu))
        self.Snew = np.tile(0.01*np.identity(self.model.ix),[self.N+1,1,1])
        self.Cnew = np.tile(0.01*np.identity(self.model.ix + self.model.iu),[self.N+1,1,1])  
        
        # line-search step size
        self.Alpha = np.power(10,np.linspace(0,-3,4))
        
        # feedforward, feedback gain
        self.l = np.zeros((self.N,self.model.iu))
        self.L = np.zeros((self.N,self.model.iu,self.model.ix))
        
        # input variance
        self.Quu_save = np.zeros((self.N,self.model.iu,self.model.iu))
        self.Quu_inv_save = np.zeros((self.N,self.model.iu,self.model.iu))
        
        # model jacobian
        self.fx = np.zeros((self.N,self.model.ix,self.model.ix))
        self.fu = np.zeros((self.N,self.model.ix,self.model.iu))
        
        # cost derivative
        self.c = np.zeros(self.N+1)
        self.cnew = np.zeros(self.N+1)
        self.cx = np.zeros((self.N+1,self.model.ix))
        self.cu = np.zeros((self.N,self.model.iu))
        self.cxx = np.zeros((self.N+1,self.model.ix,self.model.ix))
        self.cxu = np.zeros((self.N,self.model.ix,self.model.iu))
        self.cuu = np.zeros((self.N,self.model.iu,self.model.iu))
        
        # value function
        self.Vx = np.zeros((self.N+1,self.model.ix))
        self.Vxx = np.zeros((self.N+1,self.model.ix,self.model.ix))
        
        # # fitted policy
        # self.K_fit = np.zeros((self.N,self.model.iu,self.model.ix))
        # self.k_fit = np.zeros((self.N,self.model.iu))
        
    def sampleTraj(self,x0,x,u,K,Quu_inv) :
        
        x_temp = np.zeros((self.N+1,self.model.ix))
        u_temp = np.zeros((self.N,self.model.iu))
        u_mean = np.zeros((self.N,self.model.iu))
        x_temp[0,:] = x0
        for i in range(self.N) :
            u_mean[i,:] = u[i,:] + np.dot(K[i,:,:],x_temp[i,:] - x[i,:])
            u_temp[i,:] = np.random.multivariate_normal(u_mean[i,:], Quu_inv[i,:,:] )
            x_temp[i+1,:] = self.model.forwardDyn(x_temp[i,:],u_temp[i,:],i)
        
        return x_temp,u_temp
        
    def setEnv(self,policy,eta,epsilon,K_fit,k_fit,model,myAppPolicy,flagAppNN) :
    
        self.policy = policy
        self.eta = eta
        self.epsilon = epsilon
        self.model = model
        
        # fitted policy
        self.K_fit = K_fit
        self.k_fit = k_fit

        # flag
        self.flagAppNN = flagAppNN
        self.appPolicy = myAppPolicy
        
        # initial storage
        self.initStorage()
    
        
    def estimate_cost(self,x,u,eta,K_fit,k_fit,var_inv,Mu,lam):
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            K_fit = np.expand_dims(K_fit,axis=0)
            k_fit = np.expand_dims(k_fit,axis=0)
            var_inv = np.expand_dims(var_inv,axis=0)
            Mu = np.expand_dims(Mu,axis=0) # N * ic * ic
            lam = np.expand_dims(lam,axis=0) # N * ic * ic
        else :
            N = np.size(x,axis = 0)
                
        # cost function for iLQR divided by dual variable
        c1 = self.cost.estimateCost(x,u,Mu,lam) / eta
        
        # local approximated global policy
        x = np.expand_dims(x,2)
        if self.flagAppNN == False :
            k_fit = np.expand_dims(k_fit,2)
            mean_fit = np.matmul(K_fit, x) + k_fit
            mean_fit = np.squeeze(mean_fit)
        else :
            mean_fit = np.squeeze(self.appPolicy.getPolicy(np.squeeze(x,axis=2)))
        
        # cost for policy difference
        pol_diff = np.expand_dims(u - mean_fit,2)
        c2 = 0.5 * np.matmul(np.matmul(np.transpose(pol_diff,(0,2,1)),var_inv),pol_diff)

        return np.squeeze(c1 + c2)
    
    def diff_cost(self,x,u,eta,K_fit,k_fit,var_inv,Mu,lam) :
        
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            K_fit = np.expand_dims(K_fit,axis=0)
            k_fit = np.expand_dims(k_fit,axis=0)
            var_inv = np.expand_dims(var_inv,axis=0)
            Mu = np.expand_dims(Mu,axis=0) # N * ic * ic
            lam = np.expand_dims(lam,axis=0) # N * ic * ic
        else :
            N = np.size(x,axis = 0)
            
        # cost function for modified cost
        c1 = self.cost.diffCost(x,u,Mu,lam) / eta
        
        # local approximated global policy
        x = np.expand_dims(x,2)
        if self.flagAppNN == False :
            k_fit = np.expand_dims(k_fit,2)
            mean_fit = np.matmul(K_fit, x) + k_fit
            mean_fit = np.squeeze(mean_fit)
        else :
            mean_fit = np.squeeze(self.appPolicy.getPolicy(np.squeeze(x,axis=2)))
        
        pol_diff = np.expand_dims(u - mean_fit,2)
        if self.flagAppNN == False :
            pol_jacobi = K_fit
        else :
            pol_jacobi = self.appPolicy.jacobian(np.squeeze(x,axis=2))
        
        cx = np.matmul(np.matmul(np.transpose(pol_diff,(0,2,1)),var_inv), - pol_jacobi)
        cu = np.matmul(np.transpose(pol_diff,(0,2,1)),var_inv)       
        c2 = np.squeeze( np.dstack((cx,cu)) )

        return  c1 + c2
    
    def hess_cost(self,x,u,eta,K_fit,k_fit,var_inv,Mu,lam):
        
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            K_fit = np.expand_dims(K_fit,axis=0)
            k_fit = np.expand_dims(k_fit,axis=0)
            var_inv = np.expand_dims(var_inv,axis=0)
            Mu = np.expand_dims(Mu,axis=0) # N * ic * ic
            lam = np.expand_dims(lam,axis=0) # N * ic * ic
        else :
            N = np.size(x,axis = 0)
        
        # cost function for modified cost
        c1 = self.cost.hessCost(x,u,Mu,lam) / eta
        
        # local approximated global policy
        x = np.expand_dims(x,2)
        if self.flagAppNN == False :
            k_fit = np.expand_dims(k_fit,2)
            mean_fit = np.matmul(K_fit, x) + k_fit
            mean_fit = np.squeeze(mean_fit)
        else :
            mean_fit = np.squeeze(self.appPolicy.getPolicy(np.squeeze(x,axis=2)))
        # var_inv = np.tile(np.linalg.inv(var),(N,1,1))
        
        pol_diff = np.expand_dims(u - mean_fit,2)
        if self.flagAppNN == False :
            pol_jacobi = K_fit
        else :
            pol_jacobi = self.appPolicy.jacobian(np.squeeze(x,axis=2))

        # additional term
        cxx = np.matmul(np.matmul(np.transpose(- pol_jacobi,(0,2,1)),var_inv), - pol_jacobi)
        cux = np.matmul(var_inv, - pol_jacobi)
        cuu = var_inv
        
        c2 = np.squeeze(np.hstack((np.dstack((cxx,np.transpose(cux,(0,2,1)))),np.dstack((cux,cuu)))))
         
        return c1 + c2
        
    def forward(self,x0,u,K,x,k,alpha,eta,x0cov,fx,fu,Quu_inv,Mu,lam):
        # size
        ix = self.model.ix
        iu = self.model.iu
        
        # fitted policy
        K_fit = self.K_fit
        k_fit = self.k_fit
        var_inv = self.policy.policy_var_inv
        
        # horizon
        N = self.N
        
        # x-difference
        dx = np.zeros(3)
        
        # jacobian
        f = np.dstack((fx,fu))
        
        # variable setting
        xnew = np.zeros((N+1,self.model.ix))
        unew = np.zeros((N,self.model.iu))
        cnew = np.zeros(N+1)
        Snew = np.tile(0.01*np.identity(self.model.ix),[self.N+1,1,1])
        Cnew = np.tile(0.01*np.identity(self.model.ix + self.model.iu),[self.N+1,1,1]) 
        
        # initial state
        xnew[0,:] = x0
        Snew[0,:,:] = x0cov
        Cnew[0,0:ix,0:ix] = Snew[0,:,:]
        Cnew[0,0:ix,ix:ix+iu] = np.dot( Snew[0,:,:], K[0,:,:].T )
        Cnew[0,ix:ix+iu,0:ix] = np.dot( K[0,:,:], Snew[0,:,:] )
        Cnew[0,ix:ix+iu,ix:ix+iu] = np.dot( np.dot(K[0,:,:], Snew[0,:,:]), K[0,:,:].T ) + Quu_inv[0,:,:]
        Snew[1,:,:] = np.dot(np.dot(f[0,:,:],Cnew[0,:,:]),f[0,:,:].T)
        
        # roll-out
        for i in range(N):
            dx = xnew[i,:] - x[i,:]
            unew[i,:] = u[i,:] + k[i,:] * alpha + np.dot(K[i,:,:],dx)
            xnew[i+1,:] = self.model.forwardDyn(xnew[i,:],unew[i,:],i)
            cnew[i] = self.estimate_cost(xnew[i,:],unew[i,:], \
                      eta,K_fit[i,:,:],k_fit[i,:],var_inv[i,:,:],Mu[i,:,:],lam[i,:,:])
            Cnew[i,0:ix,0:ix] = Snew[i,:,:]
            Cnew[i,0:ix,ix:ix+iu] = np.dot( Snew[i,:,:], K[i,:,:].T )
            Cnew[i,ix:ix+iu,0:ix] = np.dot( K[i,:,:], Snew[i,:,:] )
            Cnew[i,ix:ix+iu,ix:ix+iu] = np.dot( np.dot(K[i,:,:], Snew[i,:,:]), K[i,:,:].T ) + Quu_inv[i,:,:]
            Snew[i+1,:,:] = np.dot(np.dot(f[i,:,:],Cnew[i,:,:]),f[i,:,:].T)
        
        cnew[N] = self.estimate_cost(xnew[N,:],np.zeros(self.model.iu),eta,
                    np.zeros((iu,ix)),np.zeros(iu),np.zeros((iu,iu)),Mu[N,:,:],lam[N,:,:])

        return xnew,unew,cnew,Snew,Cnew
        
    def backward(self):
        diverge = False
        
        # state & input size
        ix = self.model.ix
        iu = self.model.iu
        
        # V final value
        self.Vx[self.N,:] = self.cx[self.N,:]
        self.Vxx[self.N,:,:] = self.cxx[self.N,:,:]
        
        # Q function
        Qu = np.zeros(iu)
        Qx = np.zeros(ix)
        Qux = np.zeros([iu,ix])
        Quu = np.zeros([iu,iu])
        Quu_save = np.zeros([self.N,iu,iu]) # for saving
        Quu_inv_save = np.zeros([self.N,iu,iu])
        Qxx = np.zeros([ix,ix])
        
        Vxx_reg = np.zeros([ix,ix])
        Qux_reg = np.zeros([ix,iu])
        QuuF = np.zeros([iu,iu])
        
        # open-loop gain, feedback gain
        k_i = np.zeros(iu)
        K_i = np.zeros([iu,ix])
        
        self.dV[0,0] = 0.0
        self.dV[0,1] = 0.0
        
        diverge_test = 0
        for i in range(self.N-1,-1,-1):
            # print(i)
            Qu = self.cu[i,:] + np.dot(self.fu[i,:].T, self.Vx[i+1,:])
            Qx = self.cx[i,:] + np.dot(self.fx[i,:].T, self.Vx[i+1,:])
 
            Qux = self.cxu[i,:,:].T + np.dot( np.dot(self.fu[i,:,:].T, self.Vxx[i+1,:,:]),self.fx[i,:,:])
            Quu = self.cuu[i,:,:] + np.dot( np.dot(self.fu[i,:,:].T, self.Vxx[i+1,:,:]),self.fu[i,:,:])
            Qxx = self.cxx[i,:,:] + np.dot( np.dot(self.fx[i,:,:].T, self.Vxx[i+1,:,:]),self.fx[i,:,:])
            
            Vxx_reg = self.Vxx[i+1,:,:] + 0 * np.identity(ix)
            Qux_reg = self.cxu[i,:,:].T + np.dot(np.dot(self.fu[i,:,:].T, Vxx_reg), self.fx[i,:,:])
            QuuF = self.cuu[i,:,:] + np.dot(np.dot(self.fu[i,:,:].T, Vxx_reg), self.fu[i,:,:])
            Quu_save[i,:,:] = QuuF

            # add input constraints here!!
        
            # control gain      
            try:
                R = sp.linalg.cholesky(QuuF,lower=False)
            except sp.linalg.LinAlgError as err:
                diverge_test = i+1
                return diverge_test, Quu_save, Quu_inv_save
                        
            R_inv = sp.linalg.inv(R)
            QuuF_inv = np.dot(R_inv,np.transpose(R_inv))
            # Quu_inv_save[i,:,:] = np.linalg.inv(Quu)
            Quu_inv_save[i,:,:] = QuuF_inv
            k_i = - np.dot(QuuF_inv, Qu)
            K_i = - np.dot(QuuF_inv, Qux_reg)
            
            # update cost-to-go approximation
            self.dV[0,0] = np.dot(k_i.T, Qu) + self.dV[0,0]
            self.dV[0,1] = 0.5*np.dot( np.dot(k_i.T, Quu), k_i) + self.dV[0,1]
            self.Vx[i,:] = Qx + np.dot(np.dot(K_i.T,Quu),k_i) + np.dot(K_i.T,Qu) + np.dot(Qux.T,k_i)
            self.Vxx[i,:,:] = Qxx + np.dot(np.dot(K_i.T,Quu),K_i) + np.dot(K_i.T,Qux) + np.dot(Qux.T,K_i)
            self.Vxx[i,:,:] = 0.5 * ( self.Vxx[i,:,:].T + self.Vxx[i,:,:] )
                                                                                               
            # save the control gains
            self.l[i,:] = k_i
            self.L[i,:,:] = K_i
            
        return diverge_test, Quu_save, Quu_inv_save
                   
        
    def update(self,x0,u0):
        # current position
        self.x0 = x0
        
        # initial input
        self.u = u0
        
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N
        
        # initialzie parameters is already done in initializer
        self.lamda = 1e-7
        self.dlamda = 1.0
        self.lamdaFactor = 1.6
        self.dV = np.zeros((1,2))
        
        # boolian parameter setting
        diverge = False
        stop = False
        lamda_max = False
        
        # trace for iteration
        # timer, counters, constraints
        # timer begin!!
        
        # generate initial trajectory
        self.x[0,:] = self.x0
        for j in range(np.size(self.Alpha,axis=0)):
            for i in range(self.N):
                self.x[i+1,:] = self.model.forwardDyn(self.x[i,:],self.Alpha[j]*self.u[i,:],i)  
                self.c[i] = self.estimate_cost(self.x[i,:],self.Alpha[j]*self.u[i,:],self.eta,self.K_fit[i,:,:],
                            self.k_fit[i,:],self.policy.policy_var_inv[i,:,:],self.Mu_e[i,:,:],self.lam[i,:,:])
                if  np.max( self.x[i+1,:] ) > 1e8 :                
                    diverge = True
                    pass
            self.c[self.N] = self.estimate_cost(self.x[self.N,:],np.zeros(self.model.iu),
                             self.eta,np.zeros((iu,ix)),np.zeros(iu),np.zeros((iu,iu)),self.Mu_e[N,:,:],self.lam[N,:,:])
            # self.c[self.N] = 0
            if diverge == False:
                self.u = self.Alpha[j]*self.u
                break
                pass
            pass
        
        # iterations starts!!
        iteration = 0
        flgChange = True
        for iteration in range(self.maxIter) :
            # differentiate dynamics and cost
            if flgChange == True:
                start = time.clock()
                self.fx, self.fu = self.model.diffDyn(self.x[0:N,:],self.u)
                c_x_u = self.diff_cost(self.x[0:N,:],self.u,self.eta,self.K_fit,self.k_fit,
                        self.policy.policy_var_inv,self.Mu_e[0:N,:,:],self.lam[0:N,:,:])
                c_xx_uu = self.hess_cost(self.x[0:N,:],self.u,self.eta,self.K_fit,self.k_fit,self.policy.policy_var_inv,
                          self.Mu_e[0:N,:,:],self.lam[0:N,:,:])
                c_xx_uu = 0.5 * ( np.transpose(c_xx_uu,(0,2,1)) + c_xx_uu )
                self.cx[0:N,:] = c_x_u[:,0:self.model.ix]
                self.cu[0:N,:] = c_x_u[:,self.model.ix:self.model.ix+self.model.iu]
                self.cxx[0:N,:,:] = c_xx_uu[:,0:ix,0:ix]
                self.cxu[0:N,:,:] = c_xx_uu[:,0:ix,ix:(ix+iu)]
                self.cuu[0:N,:,:] = c_xx_uu[:,ix:(ix+iu),ix:(ix+iu)]
                c_x_u = self.diff_cost(self.x[N,:],np.zeros(iu),self.eta,np.zeros((iu,ix)),np.zeros(iu),np.zeros((iu,iu)),
                        self.Mu_e[N,:,:],self.lam[N,:,:])
                c_xx_uu = self.hess_cost(self.x[N,:],np.zeros(iu),self.eta,np.zeros((iu,ix)),np.zeros(iu),np.zeros((iu,iu)),
                          self.Mu_e[N,:,:],self.lam[N,:,:])
                c_xx_uu = 0.5 * ( c_xx_uu + c_xx_uu.T)
                self.cx[N,:] = c_x_u[0:self.model.ix]
                self.cxx[N,:,:] = c_xx_uu[0:ix,0:ix]
                flgChange = False
                pass
            time_derivs = (time.clock() - start)
            
            # backward pass
            backPassDone = False
            while backPassDone == False:
                start =time.clock()
                diverge,self.Quu_save,self.Quu_inv_save = self.backward()
                time_backward = time.clock() - start
                if diverge != 0 :
                    if self.verbosity == True:
                        print("Cholesky failed at %s" % (diverge))
                        pass
                    self.dlamda = np.maximum(self.dlamda * self.lamdaFactor, self.lamdaFactor)
                    self.lamda = np.maximum(self.lamda * self.dlamda,self.lamdaMin)
                    if self.lamda > self.lamdaMax :
                        break
                        pass
                    continue
                    pass
                backPassDone = True
                
            # check for termination due to small gradient
            g_norm = np.mean( np.max( np.abs(self.l) / (np.abs(self.u) + 1), axis=1 ) )
            if g_norm < self.tolGrad and self.lamda < 1e-5 :
                self.dlamda = np.minimum(self.dlamda / self.lamdaFactor, 1/self.lamdaFactor)
                if self.lamda > self.lamdaMin :
                    temp_c = 1
                    pass
                else :
                    temp_c = 0
                    pass       
                self.lamda = self.lamda * self.dlamda * temp_c 
                if self.verbosity == True:
                    print("SUCCEESS : gradient norm < tolGrad")
                    pass
                break
                pass
            

            # step3. line-search to find new control sequence, trajectory, cost
            fwdPassDone = False
            if backPassDone == True :
                start = time.clock()
                for i in self.Alpha :
                    self.xnew,self.unew,self.cnew,self.Snew,self.Cnew = self.forward(self.x0,self.u,self.L,self.x,
                            self.l,i,self.eta,self.x0cov,self.fx,self.fu,self.Quu_inv_save,self.Mu_e,self.lam)
                    dcost = np.sum( self.c ) - np.sum( self.cnew )
                    expected = -i * (self.dV[0,0] + i * self.dV[0,1])
                    if expected > 0 :
                        z = dcost / expected
                    else :
                        z = np.sign(dcost)
                        print("non-positive expected reduction: should not occur")
                        pass
                    # print(i)
                    if z > self.zMin :
                        fwdPassDone = True
                        break          
                if fwdPassDone == False :
                    alpha_temp = 1e8 # % signals failure of forward pass
                    pass
                time_forward = time.clock() - start
            # step4. accept step, draw graphics, print status 
            if self.verbosity == True and self.last_head == True:
                self.last_head = False
                # print("iteration   cost        reduction   expected    gradient    log10(lambda)")
                pass

            if fwdPassDone == True:
                if self.verbosity == True:
                    # print("%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f" % ( iteration,np.sum(self.c),dcost,expected,g_norm,np.log10(self.lamda)) )     
                    pass

                # decrese lamda
                self.dlamda = np.minimum(self.dlamda / self.lamdaFactor, 1/self.lamdaFactor)
                if self.lamda > self.lamdaMin :
                    temp_c = 1
                    pass
                else :
                    temp_c = 0
                    pass
                self.lamda = self.lamda * self.dlamda * temp_c 

                # accept changes
                self.u = self.unew
                self.x = self.xnew
                self.c = self.cnew
                self.S = self.Snew
                self.C = self.Cnew
                flgChange = True
#                 print(time_derivs)
#                 print(time_backward)
#                 print(time_forward)
                # abc
                # terminate?
                if dcost < self.tolFun :
                    if self.verbosity == True:
                        print("SUCCEESS: cost change < tolFun")
                        pass
                    break
                    pass
                
            else : # no cost improvement
                # increase lamda
                # ssprint(iteration)
                self.dlamda = np.maximum(self.dlamda * self.lamdaFactor, self.lamdaFactor)
                self.lamda = np.maximum(self.lamda * self.dlamda,self.lamdaMin)

                # print status
                if self.verbosity == True :
                    # print("%-12d%-12s%-12.3g%-12.3g%-12.3g%-12.1f" % ( iteration,'NO STEP', dcost, expected, g_norm, np.log10(self.lamda) ));
                    pass

                if self.lamda > self.lamdaMax :
                    if self.verbosity == True:
                        print("EXIT : lamda > lamdaMax")
                        pass
                    lamda_max = True
                    break
                    pass
                pass
            pass
        return self.x, self.u, self.Quu_save, self.Quu_inv_save, self.L, self.l, lamda_max
    
    def getCost(self,x,u,Mu=None,lam=None) :
        
        u_temp = np.vstack((u,np.zeros((1,self.model.iu))))
        temp_c = self.cost.estimateCost(x,u_temp,Mu,lam)
        
        return np.sum( temp_c )
        
    def getKL(self,x,u,K_fit,k_fit) :
        
        temp_KL = np.zeros(self.N)
        var = self.policy.policy_var
        var_inv = self.policy.policy_var_inv
        for i in range(self.N) :
            u_diff = np.dot(K_fit[i,:,:], x[i,:]) + k_fit[i,:] - u[i,:]
            temp_KL[i] = 0.5 * (np.trace( np.dot(var_inv[i,:,:],self.Quu_inv_save[i,:,:])) \
                            + np.dot( np.dot( np.transpose(u_diff) ,var_inv[i,:,:]), u_diff) \
                            - self.model.iu \
                            + np.log( np.linalg.det(var[i,:,:]) / np.linalg.det(self.Quu_inv_save[i,:,:])) )
        return np.sum( temp_KL )

    def iterDGD(self,x0,uIni) :
        
        u_temp = uIni
        maxIterDGD = 40
        maxIterConst = 7
        print("DGD starts!! eta = ", self.eta)
        # for j in range(maxIterConst) :
        eta_max = 1e10
        eta_min = 1e-8

        # state & input contstraints variables
        self.phi = np.ones(self.N+1) * 1.0
        # self.tolConst = 0.1

        for i in range(maxIterDGD) :
            uIni = u_temp * 1.0
            x_temp, u_temp, Quu_temp, Quu_inv_temp, K_temp, k_temp, flag_lamda = self.update(x0,uIni)
            cost = self.getCost(x_temp,u_temp)
            kl = self.getKL(x_temp,u_temp,self.K_fit,self.k_fit)
            print("cost = ", cost, "KL = ", kl, "epsilon = ", self.epsilon)

            # continuing condition
            if flag_lamda == True :
                # increase eta
                # eta_min = self.eta
                # geom = np.sqrt(eta_max * eta_min)
                # self.eta = np.minimum(10 * eta_min,geom)
                self.eta = self.eta * 2
                print("PD is not satisfied // increased eta = ", self.eta)
                continue
            else :
                pass

            # terminal condition for KL divergence
            flag_kl = False
            if kl <= 1.15 * self.epsilon and kl >= 0.85 * self.epsilon and flag_lamda == False :
                # print("======== kl satisfied =========",kl)
                flag_kl = True
            else :
                flag_kl = False
                # eta updates
                if kl < 0.85 * self.epsilon :
                    # decrease eta
                    eta_max = self.eta
                    geom = np.sqrt(eta_max * eta_min)
                    self.eta = np.maximum(0.1*eta_max,geom)
                    # self.eta = self.eta + 0.5 * (kl - self.epsilon)
                    # print("KL < epsilon // decreased eta = ", self.eta)   
                else :
                    # increase eta 
                    eta_min = self.eta
                    geom = np.sqrt(eta_max * eta_min)
                    self.eta = np.minimum(10 * eta_min,geom)
                    # self.eta = self.eta + 0.5 * (kl - self.epsilon)
                    # print("KL > epsilon // increased eta = ", self.eta)
            
            flag_c = True     
            if self.flag_const == True :
                flag_c = False
                # constraint criterion
                c_const = self.cost.ineqConst(x_temp, np.vstack((u_temp,np.zeros(self.model.iu)) )) # N * ic

                if np.max(c_const) < self.tolConst:
                    flag_c = True
                    # print("======== sc satisfied =========, max(c) = ", np.max(c_const))
                else :
                    # print("max(c) = ", np.max(c_const))
                    # Mu & lamda updates
                    # print "update lagrangian variables"
                    for i in range(self.N+1) :    
                        for j in range(self.cost.ic) :
                            if c_const[i,j] < self.phi[i] :
                                # print "Hi",c_const[i,j],i
                                self.lam[i,j,j] = np.max(( 0, self.lam[i,j,j] + self.Mu_e[i,j,j] * c_const[i,j] ))

                                # print self.lam[i,j,j]
                                self.phi[i] = self.phi[i] / 2
                            else :
                                if self.Mu_e[i,j,j] < 1e30 :
                                    self.Mu_e[i,j,j] = self.Mu_e[i,j,j] * 3
                                    # print "Hi", self.Mu_e[i,j,j]
                                else :
                                    print "Mu reaches the limit"
                    #                 pass
            else :
                if flag_kl == True :
                    # print("=================== dual gradient descent is converged ===================")
                    # print("eta = ", self.eta)
                    break
                else :
                    continue
                    pass

            if flag_c == True and flag_kl == True :
                print("=================== dual gradient descent is converged ===================")
                print("eta = ", self.eta)
                self.tolConst = self.tolConst * 0.9
                print("EXIT : max(c)", np.max(c_const), " < tolConst, tolConst becomes, ", self.tolConst)
                print "update lagrangian variables"
                for i in range(self.N+1) :    
                    for j in range(self.cost.ic) :
                        if c_const[i,j] < self.phi[i] :
                            # print "Hi",c_const[i,j],i
                            self.lam[i,j,j] = np.max(( 0, self.lam[i,j,j] + self.Mu_e[i,j,j] * c_const[i,j] ))

                            # print self.lam[i,j,j]
                            self.phi[i] = self.phi[i] / 2
                        else :
                            if self.Mu_e[i,j,j] < 1e30 :
                                self.Mu_e[i,j,j] = self.Mu_e[i,j,j] * 3
                                # print "Hi", self.Mu_e[i,j,j]
                            else :
                                print "Mu reaches the limit"
                #                 pass
                break
            else :
                pass

                
        return x_temp, u_temp, Quu_temp, Quu_inv_temp, self.eta, cost, K_temp, k_temp
        
        



