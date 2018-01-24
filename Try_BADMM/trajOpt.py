
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


# In[47]:

class trajOpt:
    def __init__(self,name,horizon):
        # class name
        self.name = name
        
        # given policy
        # self.policy = policy
        
        # model & cost
        self.model = model.unicycle('paul')
        self.cost = cost.unicycle('john')
        
             
        # input & output size
        self.ix = self.model.ix
        self.iu = self.model.iu

        # iLQR parameters
        self.verbosity = True
        self.dlamda = 1.0
        self.lamda = 1.0
        self.lamdaFactor = 1.6
        self.lamdaMax = 1e10
        self.lamdaMin = 1e-6
        self.tolFun = 1e-2
        self.tolGrad = 1e-4
        self.maxIter = 5000
        self.zMin = 0
        self.last_head = True
        self.dV = np.zeros((1,2))
        self.N = horizon
        
        # dual gradient descent variable
        self.nu = 1 * np.ones(self.N) * 1
        self.dual = np.ones((self.N,self.model.iu)) * 0
                
        # current policy network (temp) / it should be replaced with the result from polOpt
        self.pol_mu = np.zeros(self.model.iu)
        self.pol_var = np.tile(np.identity(self.model.iu),[self.N,1,1])
        
        # initial trajectory
        self.x0 = np.zeros(self.model.ix)
#         self.x0var = np.identity(self.model.ix)
        self.x = np.zeros((self.N+1,self.model.ix))
#         self.S = np.tile(0.01*np.identity(self.model.ix),[self.N+1,1,1])
        self.u = np.zeros((self.N,self.model.iu))
#         self.A = np.tile(np.identity(self.model.iu),[self.N,1,1])
#         self.sigma = np.tile(0.01*np.identity(self.model.ix+
#                                             self.mode.iu),[self.N+1,1,1])
        
        # next trajectroy
        self.xnew = np.zeros((self.N+1,self.model.ix))
        self.unew = np.zeros((self.N,self.model.iu))
        
        # for line-search
        # self.Alpha = np.zeros(4)
        self.Alpha = np.power(10,np.linspace(0,-3,4))
        
        # feedforward, feedback gain
        self.l = np.zeros((self.N,self.model.iu))
        self.L = np.zeros((self.N,self.model.iu,self.model.ix))
        
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
        
    def setEnv(self,policy,dual,nu) :
    
        self.policy = policy
        self.dual = dual
        self.nu = nu
        
    def estimate_cost(self,x,u,nu,dual):
                
        # cost function for iLQR
        c1 = self.cost.estimateCost(x,u)
#         c1 = np.divide(self.cost.estimateCost(x,u), nu)
        
        # cost for maximum entropy control
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            c2 = - np.vdot(u,dual)
        else :
            dual = np.squeeze(dual)
            c2 = - np.einsum('ij,ij->j', u.T, dual.T)
        
        c12 = np.divide(c1+c2,nu)
        # print_np( self.policy.gaussPDF(x,u) )
        c3 = - np.log( self.policy.gaussPDF(x,u) )
        
        # divergence
        return c12 + c3;
    
    def diff_cost(self,x,u,nu,dual) :
        
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            dual = np.expand_dims(dual,axis=0)
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
        
        nu_aug = np.repeat(nu,(ix+iu),axis=0)
        dual_aug = np.repeat(dual,(ix+iu),axis=0)

        # numerical difference
        c_nominal = self.estimate_cost(x,u,nu,dual)
        c_change = self.estimate_cost(x_aug,u_aug,nu_aug,dual_aug)
        c_change = np.reshape(c_change,(N,1,iu+ix))

        c_diff = ( c_change - np.reshape(c_nominal,(N,1,1)) ) / h
        c_diff = np.reshape(c_diff,(N,iu+ix))
            
        return  np.squeeze(c_diff)
    
    def hess_cost(self,x,u,nu,dual):
        
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            dual = np.expand_dims(dual,axis=0)
        else :
            N = np.size(x,axis = 0)
#             nu = np.reshape(nu,(N,1))
            
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
        
        nu_aug = np.repeat(nu,(ix+iu),axis=0)
        dual_aug = np.repeat(dual,(ix+iu),axis=0)


        # numerical difference
        c_nominal = self.diff_cost(x,u,nu,dual)
        c_change = self.diff_cost(x_aug,u_aug,nu_aug,dual_aug)
        c_change = np.reshape(c_change,(N,iu+ix,iu+ix))
        c_hess = ( c_change - np.reshape(c_nominal,(N,1,ix+iu)) ) / h
        c_hess = np.reshape(c_hess,(N,iu+ix,iu+ix))
         
        return np.squeeze(c_hess)
        
        
    def forward(self,x0,u,K,x,k,alpha,nu,dual,x0cov,fx,fu,Quu_inv):
        # size
        ix = self.model.ix
        iu = self.model.iu
        
        # horizon
        N = self.N
        
        # x-difference
        dx = np.zeros(3)
        
        # jacobian
        f = np.hstack((fx,fu))
        
        # variable setting
        xnew = np.zeros((N+1,self.model.ix))
        unew = np.zeros((N,self.model.iu))
        cnew = np.zeros(N+1)
        Snew = np.tile(0.01*np.identity(self.model.ix),[self.N+1,1,1])
        Cnew = np.tile(0.01*np.identity(self.model.ix + self.mode.iu),[self.N+1,1,1]) 
        
        # initial state
        xnew[0,:] = x0
        Snew[0,:,:] = x0cov
        Cnew[0,0:ix,0:ix] = Snew[0,:,:]
        Cnew[0,0:ix,ix:ix+iu] = np.dot( Snew[0,:,:], K[0,:,:].T )
        Cnew[0,ix:ix+iu,0:u] = np.dot( K[0,:,:], Snew[0,:,:] )
        Cnew[0,ix:ix+iu,ix:ix+iu] = np.dot( np.dot(K[0,:,:].T, Snew[0,:,:]), K[0,:,:].T ) + Quu_inv[0,:,:]
        Snew[1,:,:] = np.dot(np.dot(f[0,:,;],Cnew[0,:,:]),f[0,:,:].T)
        
        # roll-out
        for i in range(N):
            dx = xnew[i,:] - x[i,:]
            unew[i,:] = u[i,:] + k[i,:] * alpha + np.dot(K[i,:,:],dx)
            xnew[i+1,:] = self.model.forwardDyn(xnew[i,:],unew[i,:])
            cnew[i] = self.estimate_cost(xnew[i,:],unew[i,:],nu[i],dual[i,:])
            Cnew[i,0:ix,0:ix] = Snew[i,:,:]
            Cnew[i,0:ix,ix:ix+iu] = np.dot( Snew[i,:,:], K[i,:,:].T )
            Cnew[i,ix:ix+iu,0:u] = np.dot( K[i,:,:], Snew[i,:,:] )
            Cnew[i,ix:ix+iu,ix:ix+iu] = np.dot( np.dot(K[i,:,:].T, Snew[i,:,:]), K[i,:,:].T ) + Quu_inv[i,:,:]
            Snew[i+1,:,:] = np.dot(np.dot(f[i,:,;],Cnew[i,:,:]),f[i,:,:].T)
        
        cnew[N] = self.estimate_cost(xnew[N,:],np.zeros(self.model.iu),np.mean(nu),np.zeros(self.model.iu))
        # cnew[N] = 0
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
            
            Vxx_reg = self.Vxx[i+1,:,:] + self.lamda * np.identity(ix)
            Qux_reg = self.cxu[i,:,:].T + np.dot(np.dot(self.fu[i,:,:].T, Vxx_reg), self.fx[i,:,:])
            QuuF = self.cuu[i,:,:] + np.dot(np.dot(self.fu[i,:,:].T, Vxx_reg), self.fu[i,:,:])
            Quu_save[i,:,:] = Quu
            # add input constraints here!!
        
            
            # control gain      
            try:
                R = sp.linalg.cholesky(QuuF,lower=False)
            except sp.linalg.LinAlgError as err:
                diverge_test = i+1
                return diverge_test, Quu_save, Quu_inv_save
                        
            R_inv = sp.linalg.inv(R)
            QuuF_inv = np.dot(R_inv,np.transpose(R_inv))
            Quu_inv_save[i,:,:] = np.linalg.inv(Quu)
            # Quu_inv_save[i,:,:] = QuuF_inv
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
        self.x0 = x0;
        
        # initial input
        self.u = u0;
        
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N
        
        # timer setting
        
        # initialzie parameters
        self.lamda = 1.0;
        self.dlamda = 1.0;
        self.lamdaFactor = 1.6
        
        # boolian parameter setting
        diverge = False
        stop = False
        
        # trace for iteration
        # timer, counters, constraints
        # timer begin!!
        
        # generate initial trajectory
        self.x[0,:] = self.x0
        for j in range(np.size(self.Alpha,axis=0)):
            for i in range(self.N):
                self.x[i+1,:] = self.model.forwardDyn(self.x[i,:],self.Alpha[j]*self.u[i,:])  
                self.c[i] = self.estimate_cost(self.x[i,:],self.Alpha[j]*self.u[i,:],self.nu[i],self.dual[i,:])
                if  np.max( self.x[i+1,:] ) > 1e8 :                
                    diverge = True
                    pass
            self.c[self.N] = self.estimate_cost(self.x[self.N,:],np.zeros(self.model.iu),np.mean(self.nu),np.zeros(self.model.iu))
            # self.c[self.N] = 0
            if diverge == False:
                break;
                pass
            pass
        
        # iterations starts!!
        iteration = 0
        flgChange = True
        for iteration in range(self.maxIter) :
            # differentiate dynamics and cost
            # TODO - we need a more fast algorithm to calculate derivs of dyn, cost
            #        1. using not for loop, but elementwise calculation
            if flgChange == True:
                start = time.clock()
                self.fx, self.fu = self.model.diffDyn(self.x[0:N,:],self.u)
                c_x_u = self.diff_cost(self.x[0:N,:],self.u,self.nu,self.dual)
                c_xx_uu = self.hess_cost(self.x[0:N,:],self.u,self.nu,self.dual)
                c_xx_uu = 0.5 * ( np.transpose(c_xx_uu,(0,2,1)) + c_xx_uu )
                self.cx[0:N,:] = c_x_u[:,0:self.model.ix]
                self.cu[0:N,:] = c_x_u[:,self.model.ix:self.model.ix+self.model.iu]
                self.cxx[0:N,:,:] = c_xx_uu[:,0:ix,0:ix]
                self.cxu[0:N,:,:] = c_xx_uu[:,0:ix,ix:(ix+iu)]
                self.cuu[0:N,:,:] = c_xx_uu[:,ix:(ix+iu),ix:(ix+iu)]
                c_x_u = self.diff_cost(self.x[N,:],np.zeros(iu),np.mean(self.nu),np.zeros(iu))
                c_xx_uu = self.hess_cost(self.x[N,:],np.zeros(iu),np.mean(self.nu),np.zeros(iu))
                # c_x_u = self.diff_cost(np.zeros(ix),np.zeros(iu))
                # c_xx_uu = self.hess_cost(np.zeros(ix),np.zeros(iu))
                c_xx_uu = 0.5 * ( c_xx_uu + c_xx_uu.T)
                self.cx[N,:] = c_x_u[0:self.model.ix]
                self.cxx[N,:,:] = c_xx_uu[0:ix,0:ix]
                flgChange = False
                pass
            time_derivs = (time.clock() - start)
            
            # backward pass
            backPassDone = False;
            while backPassDone == False:
                start =time.clock()
                diverge,Quu_save,Quu_inv_save = self.backward()
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
                    self.xnew,self.unew,self.cnew = self.forward(self.x0,self.u,self.L,self.x,self.l,i,self.nu,self.dual)
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
                print("iteration   cost        reduction   expected    gradient    log10(lambda)")
                pass

            if fwdPassDone == True:
                if self.verbosity == True:
                    print("%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f" % ( iteration,np.sum(self.c),dcost,expected,g_norm,np.log10(self.lamda)) )     
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
                    print("%-12d%-12s%-12.3g%-12.3g%-12.3g%-12.1f" %
                     ( iteration,'NO STEP', dcost, expected, g_norm, np.log10(self.lamda) ));
                    pass

                if self.lamda > self.lamdaMax :
                    if self.verbosity == True:
                        print("EXIT : lamda > lamdaMax")
                        pass
                    break
                    pass
                pass
            pass
        
        return self.x, self.u, Quu_save, Quu_inv_save


# In[48]:

# i1 = trajOpt('doublePendulum',200)
# x0 = np.zeros(4)
# u0 = np.ones( (200,2) )
# x0[0] = -np.pi/2
# x0[1] = 0.
# x,u, Quu_save, Quu_inv_save = i1.update(x0,u0)

