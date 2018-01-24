
# coding: utf-8

# In[ ]:

from __future__ import division
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
import tensorflow as tf
from termcolor import colored
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))


# In[ ]:

import model
import cost
import iLQR
import poliOpt
import trajOpt
import policyClass
import modelLearn
from gps_util import getCostNN, getCostAppNN, getCostTraj, getObs, getDesired

class mdGPS :
    def __init__(self,name):
        # class name
        self.name = name
        
        # size of state, input, horizon, hidden node
        self.ix = 3
        self.iu = 2
        self.io = 3
        self.N = 150
        self.hidden_num = 40
        
                
        # the number of initial position 
        self.num_ini = 3
        self.x0 = np.array( [ [0,-2,np.pi/2],[2,-2,np.pi/2],[-2,-2,np.pi/2]] )
        # self.x0 = np.array( [ [0,-2,np.pi/2]] )
        self.u0 = np.zeros( (self.N,2) )
        self.x_t = np.array([0,3])
        
        self.stepIni = 0.02
        
        # policy, traj class, real-robot model
        self.myModel = model.unicycle('unicycle')
        self.myCost = cost.unicycle('unicycle',self.x_t)
        self.myPolicy = poliOpt.poliOpt('unicycle',self.hidden_num,5000,self.stepIni,self.io,self.iu,self.N)
        self.myTraj = trajOpt.trajOpt('unicycle',self.N,self.myModel,self.myCost)
        
        # step size used in step adjustment
        self.epsilon = 5
        self.eps_max = 1000
        self.eps_min = 0.5
        
        # maxIter used for update function
        self.maxIter = 10
        
        # the number of sample for fitted model, which determines how many times robot actually works
        self.num_fit = 20
        
        # the number of sample for policy optimization / samples are drawn from the local policy from iLQR
        self.num_sample = self.num_fit
        
        # NN iter
        self.iter_NN = 100
        
        # policy type
        self.onPolicy = True
        
        # observation?
        self.obsFlag = True

    def update(self) :
        #
        N = self.N
        ix = self.ix
        iu = self.iu
        io = self.io
        
        # save variable
        W1_save = np.zeros((self.maxIter+1,self.ix,self.hidden_num))
        W2_save = np.zeros((self.maxIter+1,self.hidden_num,self.hidden_num))
        W3_save = np.zeros((self.maxIter+1,self.hidden_num,self.iu))
        b1_save = np.zeros((self.maxIter+1,self.hidden_num))
        b2_save = np.zeros((self.maxIter+1,self.hidden_num))
        b3_save = np.zeros((self.maxIter+1,self.iu))
        var_save = np.zeros((self.maxIter+1,self.N,self.iu,self.iu))
        
        # stepsize setting for supervised learning
        stepIni = self.stepIni
        stepSize = stepIni
           
        # dual variable
        eta_ini = 1
        eta = eta_ini
        etaPI_ini = 1 * np.ones(N)
        etaPI = etaPI_ini
        
        # initial cost
        cNmP = 1e8
        cNmN = 1e8
        cNmN_pre = 1e8
        cPmP_NN = 1e8 
        cNmN_NN = 1e8
        
        # real cost from robot
        cost_real = np.zeros(self.maxIter)
        
        # Initial policy from iLQR under known nominal model (maybe unicycle..)
        W1,b1,W2,b2,W3,b3,pol_var,pol_var_inv,costNN_pre,uNominal, iniPolicy = self.getInitialPolicy(stepIni)
        
        # parameter save
        W1_save[0,:,:] = W1
        W2_save[0,:,:] = W2
        W3_save[0,:,:] = W3
        b1_save[0,:] = b1
        b2_save[0,:] = b2
        b3_save[0,:] = b3
        var_save[0,:,:,:] = pol_var
        
        # initialize local policy, local approximation to the global policy
        localPolicySet = [''] * self.num_ini
        appGlobalPolicySet = [''] * self.num_ini
        
        # initialize fitted model
        myFitModelSet = [''] * self.num_ini
        for j in range(self.num_ini) :
            myFitModelSet[j] = modelLearn.modelLearn('hi',self.ix,self.iu,self.N,10,np.ones((N,8,1)))
        myFitModelOldSet = myFitModelSet
        
        # initialize fitted model
        myFitPolicySet = [''] * self.num_ini
        for j in range(self.num_ini) :
            myFitPolicySet[j] = modelLearn.policyLearn('hi',self.ix,self.iu,self.N,10,np.ones((N,5,1)))

        
        ####################### Variables ###########################
        # x_save ---> data for supervised learning of policy
        # x_new, x_traj  ---> the result of trajectory optimization along initial position
        # x_fit ---> data to fit linearized policy
        # x_ini  ---> data for initial policy opitmization, it is resutled from iLQR.
        # space for trajectory from optimization along initial condition
        x_traj = np.zeros((self.num_ini,self.N+1,self.ix))
        u_traj = np.zeros((self.num_ini,self.N,self.iu))
        K_traj = np.zeros((self.num_ini,self.N,self.iu,self.ix))
        Quu_inv_traj = np.zeros((self.num_ini,self.N,self.iu,self.iu))
        
        # flag
        print("======================== Mirror descent GPS ============================ ")
        for i in range(self.maxIter) :
            print colored('iteration = ', 'blue'), colored(i+1, 'blue')
            print("trajectory optimization start!!")
            
            if i > 0 :
                self.num_fit = 20
                self.num_sample = self.num_fit

            # data used for supervised learning
            x_save = np.zeros((N*self.num_ini*self.num_sample,ix))
            u_save = np.zeros((N*self.num_ini*self.num_sample,iu))
            o_save = np.zeros((N*self.num_ini*self.num_sample,io))
            var_inv_save = np.zeros((N*self.num_ini*self.num_sample,iu,iu))
            var_inv_set_save = np.zeros((N,iu,iu,self.num_ini*self.num_sample))
            cost_save_cm = np.zeros(self.num_ini)
            cost_save_pm = np.zeros(self.num_ini)
            eta_save = np.zeros(self.num_ini)       
            
            cost_real_ini = np.zeros(self.num_ini)
            for j in range(self.num_ini) :
                print(j, "initial position in trajectory optimization")
                # initial position selection
                # x_0 = self.x0[j,:]
                
                # 1. move real robot to get sample used for model learning, approximate global policy
                if i == 0:
                    x_fit, u_fit_m, cost_fit, o_fit = self.driveRobot(self.x0[j,:],self.onPolicy,i,iniPolicy) 
                else :
                    x_fit, u_fit_m, cost_fit, o_fit = self.driveRobot(self.x0[j,:],self.onPolicy,i,localPolicySet[j])
                cost_real_ini[j] = np.mean(cost_fit)
                
                
                # 2. fit linear gaussian global dynamics
                myFitModelOldSet[j] = myFitModelSet[j]
                print("fit model prior !!")
                myFitModelSet[j].update(x_fit,u_fit_m)
                if i == 0 : 
                    myFitModelSet[j].data_aug(x_fit,u_fit_m,True)
                else : 
                    myFitModelSet[j].data_aug(x_fit,u_fit_m)
                
                # 3. fit linearized global policy using samples
                # this fitted policy should be class
                appGlobalPolicySet[j], myFitPolicySet[j] = self.fittingGlobalPolicy(x_fit,o_fit,myFitPolicySet[j])
                K_fit = appGlobalPolicySet[j].K_mat
                k_fit = appGlobalPolicySet[j].k_mat
                if i == 0 :
                    myFitPolicySet[j].data_aug(x_fit,appGlobalPolicySet[j].u_nominal,True)
                else : 
                    myFitPolicySet[j].data_aug(x_fit,appGlobalPolicySet[j].u_nominal)
                
                
                
                # 4. trajectory optimization 
                self.myTraj.setEnv(self.myPolicy,eta,self.epsilon,K_fit,k_fit,myFitModelSet[j])
                x_new,u_new, Quu_new, Quu_inv_new, eta_new, cost_new, K_new, k_new = self.myTraj.iterDGD(self.x0[j,:],uNominal)
                localPolicySet[j] = policyClass.policy("local",ix,iu,N)
                localPolicySet[j].setter(K_new,k_new,x_new,u_new,Quu_inv_new)
                
                # cost, eta save
                eta_save[j] = eta_new
                cost_save_cm[j] = getCostTraj(self.x0[j,:],localPolicySet[j],N,self.myTraj,myFitModelSet[j])
                cost_save_pm[j] = getCostTraj(self.x0[j,:],localPolicySet[j],N,self.myTraj,myFitModelOldSet[j])
                # new setting
                eta = eta_new
#                 uNominal = u_new 

                # 5. generate samples for supervised learning
                num_sample = self.num_sample
                for k in range(num_sample):
                    # sample extraction
                    x_sample = x_fit[:,:,k]
                    u_sample = np.zeros((N,iu))
                    u_sample = u_new + k_new + np.squeeze( np.matmul(K_new,np.expand_dims(x_sample[0:N,:]-x_new[0:N,:],axis=2)))
                    o_sample = o_fit[:,:,k]

                    # data for supervised learning
                    index = j * num_sample + k
                    x_save[N*index:N*(index+1),:] = x_sample[0:N,:]
                    u_save[N*index:N*(index+1),:] = u_sample
                    o_save[N*index:N*(index+1),:] = o_sample[0:N,:]
                    var_inv_save[N*index:N*(index+1),:,:] = Quu_new
                    var_inv_set_save[:,:,:,index] = Quu_new

            # get cost
            cNmN = np.mean(cost_save_cm)
            cNmP = np.mean(cost_save_pm)
            # set eta
            eta = eta_ini
            print colored('trajectory optimization is finished Traj estimated cost is', 'blue'), colored(cNmN, 'blue')

            # 6. policy optimzation using supervised learning based on SGD
            print("policy optimization start!!")
            stepSize = stepSize * 0.6
            self.myPolicy.setEnv(self.iter_NN,stepIni)
            for jj in range(20) :               
                W1,b1,W2,b2,W3,b3,pol_var,pol_var_inv = self.myPolicy.update(o_save,u_save,var_inv_save * np.mean(eta_save) ,var_inv_set_save,self.num_ini*self.num_sample)
                c_temp = np.zeros(self.num_ini)
                for j in range(self.num_ini) :
                    c_temp[j] = getCostAppNN(self.x0[j,:],appGlobalPolicySet[j],N,self.myTraj,myFitModelSet[j])
                    
                if np.mean(c_temp) >= 20 * cNmN or np.isnan(np.mean(c_temp)) :
                    print colored('policy optimization is diversed cost is', 'red'), colored(np.mean(c_temp), 'red')
                    # flag_eps = False
                    stepSize = stepIni
                    self.myPolicy.setEnv(self.iter_NN, stepSize / (10*jj+1))
                    # epsilon = 0.5 * epsilon
                else : 
                    cPmP_NN = cNmN_NN
                    cNmN_NN = np.mean(c_temp)
                    print colored('policy optimization is finished NN estimated cost is', 'red'), colored(cNmN_NN, 'red')
                    break

            # 6. step size adjustment epsilon setting
            # cNmP     ----> cost based on current policy and previous fitted model
            # cNmN     ----> cost based on current policy and current fitted model
            # cPmP_NN  ----> cost based on previous NN policy and previous fitted model
            # cNmN_NN  ----> cost based on current NN policy and current fitted model 
            if i != 0 :
                self.epsilon = self.epsilon * (cNmP - cPmP_NN) / ( 2 * (cNmP - cNmN_NN))
                print("epsilon is", self.epsilon)
                if self.epsilon <= 0 :
                    self.epsilon = 20
                    print("epsilon becomes minus.. I change epsilon as", self.epsilon)
                else : 
                    if self.epsilon > self.eps_max :
                        self.epsilon = self.eps_max
                    elif self.epsilon < self.eps_min :
                        self.epsilon = self.eps_min
            
            # parameter save
            W1_save[i+1,:,:] = W1
            W2_save[i+1,:,:] = W2
            W3_save[i+1,:,:] = W3
            b1_save[i+1,:] = b1
            b2_save[i+1,:] = b2
            b3_save[i+1,:] = b3
            var_save[i+1,:,:,:] = pol_var
            
            # cost save
            cost_real[i] = np.mean(cost_real_ini)
            
            # test policy
            self.testRobot(self.x0) 
            # 7. terminal condition (when the change of cNmN is extremly low)            
            if np.abs(cNmN_NN - cPmP_NN) < 0.5 :
                print("SUCCEESS : cost change < tolFun")
                break
            else :
                cNmN_pre = cNmN
            
                
        return W1_save, W2_save, W3_save, b1_save, b2_save, b3_save, var_save,i,cost_real
          
    def testRobot(self,x0) :
        print "test starts"
        
        # parameter
        N = self.N
        ix = self.ix
        iu = self.iu
        io = self.io
        num_fit = 3
        
        # data used for model learning
        x_fit = np.zeros((N+1,ix,num_fit))
        o_fit = np.zeros_like(x_fit)
        u_fit_m = np.zeros((N,iu,num_fit))
        cost_fit = np.zeros(num_fit)
        
        for im in range(num_fit):
            x_fit[0,:,im] = x0[im,:] + 0 * np.array((1,1,0.3)) * (np.random.random(3) - 0.5)
            o_fit[0,:,im] = getObs(self.x_t,x_fit[0,:,im],self.obsFlag)
            
            for ip in range(N) :

                # mean policy will be performed    
                '''
                if ip == 0 :
                    o_temp = np.vstack((o_fit[ip,:,im],o_fit[ip,:,im]))    
                else :
                    o_temp = np.vstack((o_fit[ip-1,:,im],o_fit[ip,:,im]))
                '''
                o_temp = o_fit[ip,:,im]
                u_temp, var_temp = self.myPolicy.getPolicy(o_temp)
                u_temp = np.squeeze(u_temp)  
                      
                u_fit_m[ip,:,im] = u_temp
                x_fit[ip+1,:,im] = self.myModel.forwardDyn(x_fit[ip,:,im], u_fit_m[ip,:,im], ip)
                o_fit[ip+1,:,im] = getObs(self.x_t,x_fit[ip+1,:,im],self.obsFlag)
            cost_fit[im] = self.myTraj.getCost(x_fit[:,:,im],u_fit_m[:,:,im])
            
            print colored('mean value of real cost is', 'yellow'), colored(cost_fit[im], 'yellow')
        plt.figure(1)    
        plt.subplot(121)
        plt.axis([-3, 3, -3, 3])
        for im in range(num_fit) : 
            plt.plot(x_fit[:,0,im],x_fit[:,1,im])
        plt.subplot(122)
        for im in range(num_fit) : 
            plt.plot(range(0,N),u_fit_m[:,0,im])
        plt.show()   
                
    def fittingGlobalPolicy(self,x_fit,o_fit,myFitPolicy) :
        # variables
        N = self.N
        ix = self.ix
        iu = self.iu
        io = self.io
        num_fit = self.num_fit
        
        print("fit linearized global policy")
        u_fit_p = np.zeros((N,iu,num_fit))
        K_fit = np.zeros((N,iu,ix))
        k_fit = np.zeros((N,iu))
        for ip in range(N) :
            for im in range(num_fit) :
                # sample from global policy
                '''
                if ip == 0 :
                    o_temp = np.vstack((o_fit[ip,:,im],o_fit[ip,:,im]))    
                else :
                    o_temp = np.vstack((o_fit[ip-1,:,im],o_fit[ip,:,im]))
                '''
                o_temp = o_fit[ip,:,im]     
                u_fit_p[ip,:,im], var_temp = self.myPolicy.getPolicy(o_temp)
            
        myFitPolicy.update(x_fit, u_fit_p)
#             # linear regression
#             xMat_fit = np.vstack(( x_fit[ip,:,:] , np.ones((1,num_fit)) ))
#             uMat_fit = u_fit_p[ip,:,:]
#             KMat_fit = np.dot(uMat_fit, np.linalg.pinv(xMat_fit))
#             K_fit[ip,:,:] = KMat_fit[:,0:ix]
#             k_fit[ip,:] = KMat_fit[:,ix]
            
        tempPolicy = policyClass.policy("global",ix,iu,N)
        tempPolicy.setter(myFitPolicy.K,myFitPolicy.k,x_fit,u_fit_p,var_temp)
            
        return tempPolicy, myFitPolicy
                                   
    def driveRobot(self,x0,onPolicy,mainIter,localPolicy) :
        
        # local policy
        x_traj = localPolicy.x_nominal
        u_traj = localPolicy.u_nominal
        k_traj = localPolicy.k_mat
        K_traj = localPolicy.K_mat
        Quu_inv_traj = localPolicy.polVar
        
        # parameter
        N = self.N
        ix = self.ix
        iu = self.iu
        io = self.io
        num_fit = self.num_fit
        
        # data used for model learning
        x_fit = np.zeros((N+1,ix,num_fit))
        o_fit = np.zeros_like(x_fit)
        u_fit_m = np.zeros((N,iu,num_fit))
        cost_fit = np.zeros(num_fit)
        
        for im in range(num_fit):
            x_fit[0,:,im] = x0 + np.array((1,1,0.3)) * (np.random.random(3) - 0.5)
            o_fit[0,:,im] = getObs(self.x_t,x_fit[0,:,im],self.obsFlag)
            
            for ip in range(N) :
                if onPolicy == False :
                    
                    # u_temp = u_traj[ip,:] + k_traj[ip,:] + np.dot(K_traj[ip,:,:],x_fit[ip,:,im] - x_traj[ip,:])
                    u_temp = np.random.multivariate_normal(u_traj[ip,:] + k_traj[ip,:] + np.dot(K_traj[ip,:,:],x_fit[ip,:,im] - x_traj[ip,:]),Quu_inv_traj[ip,:,:] / 1 )
                      
                else :
                    
                    '''
                    if ip == 0 :
                        o_temp = np.vstack((o_fit[ip,:,im],o_fit[ip,:,im]))    
                    else :
                        o_temp = np.vstack((o_fit[ip-1,:,im],o_fit[ip,:,im]))
                    '''
                    o_temp = o_fit[ip,:,im]     
                    u_temp, var_temp = self.myPolicy.getPolicy(o_temp)
                    u_temp = np.random.multivariate_normal(np.squeeze(u_temp), var_temp[ip,:,:] / 1 )
                    u_temp = np.squeeze(u_temp)               

                u_fit_m[ip,:,im] = u_temp
                x_fit[ip+1,:,im] = self.myModel.forwardDyn(x_fit[ip,:,im], u_fit_m[ip,:,im], ip)
                o_fit[ip+1,:,im] = getObs(self.x_t,x_fit[ip+1,:,im],self.obsFlag)
            cost_fit[im] = self.myTraj.getCost(x_fit[:,:,im],u_fit_m[:,:,im])
            
            print colored('mean value of real cost is', 'yellow'), colored(cost_fit[im], 'yellow')
        plt.figure(1)    
        plt.subplot(121)
        plt.axis([-3, 3, -3, 3])
        for im in range(num_fit) : 
            plt.plot(x_fit[:,0,im],x_fit[:,1,im])
        plt.subplot(122)
        for im in range(num_fit) : 
            plt.plot(range(0,N),u_fit_m[:,0,im])
        plt.show()
        return x_fit, u_fit_m, cost_fit, o_fit
                
    def getInitialPolicy(self,stepIni) :
        
        print("initial trajectories from iLQR")  
        # Initial trajectory distribution from iLQR with low number of iterations
        i1 = iLQR.iLQR('unicycle',self.N,2,self.myModel,self.myCost)
        x0 = self.x0
        u0 = self.u0
        num_sample = 20
        N = self.N
        ix = self.ix
        iu = self.iu
        io = self.io
        
        # space for trajectory saving
        x_ini = np.zeros((N*num_sample,ix))
        u_ini = np.zeros((N*num_sample,iu))
        o_ini = np.zeros((N*num_sample,io))
        var_inv_ini = np.zeros((N*num_sample,iu,iu))
        var_inv_set_ini = np.zeros((N,iu,iu,num_sample))

        # iLQR with prior model
        x, u, Quu, Quu_inv, K, k_mat = i1.update(x0[0,:],u0)
        u = u / 4
        K = K
        k_mat = k_mat / 4

        for k in range(num_sample):
            x0_sample = x0[0,:] + np.array((1,1,0.3)) * ( np.random.random(3) - 0.5 )

            # sample extraction
            x_sample, u_sample = self.myTraj.sampleTraj(x0_sample,x,u,K,Quu_inv)
            c_sample = self.myTraj.getCost(x_sample,u_sample)
            print("sample generation cost = ", c_sample)
            
            # get observation // just for simulation. this thing never happens in real..
            o_sample = np.zeros_like(x_sample)
            for ip in range(N+1) :
                o_sample[ip,:] = getObs(self.x_t,x_sample[ip,:],self.obsFlag)

            # data for supervised learning
            index = k
            x_ini[N*index:N*(index+1),:] = x_sample[0:N,:]
            u_ini[N*index:N*(index+1),:] = u_sample
            o_ini[N*index:N*(index+1),:] = o_sample[0:N,:]
            var_inv_ini[N*index:N*(index+1),:,:] = Quu 
            var_inv_set_ini[:,:,:,k] = Quu
        
        iniPolicy = policyClass.policy("local",ix,iu,N)
        
        print("initial policy optimization starts!!")
        for j in range(5) :
            self.myPolicy.setEnv(self.iter_NN,stepIni / (j+1) )
            W1,b1,W2,b2,W3,b3,pol_var,pol_var_inv = self.myPolicy.update(o_ini,u_ini,var_inv_ini,var_inv_set_ini,num_sample)
            # costNN_pre = getCostNN(x0[0,:],self.myPolicy,N,self.myTraj,self.myModel)
            costNN_pre = 1e4
            if costNN_pre < 1e10 :
#                 cPmP_NN = costNN_pre
                print colored('initial policy parameter is updated true cost is ', 'green'), colored(costNN_pre, 'green')
                break
            else : 
                print("NN is diversed")
                pass
            pass
        
        iniPolicy.setter(K,k_mat,x,u , pol_var)
        
        return W1,b1,W2,b2,W3,b3,pol_var,pol_var_inv,costNN_pre,u, iniPolicy
        # return pol_var,pol_var_inv,costNN_pre,u, iniPolicy


# In[ ]:

# myGPS = mdGPS('hi')
# W1_save, W2_save, W3_save, b1_save, b2_save, b3_save, var_save, i ,cost_real = myGPS.update()


# In[ ]:

# # i=4
# np.savetxt('w1.txt',W1_save[i,:,:])
# np.savetxt('w2.txt',W2_save[i,:,:])
# np.savetxt('w3.txt',W3_save[i,:,:])
# np.savetxt('b1.txt',b1_save[i,:])
# np.savetxt('b2.txt',b2_save[i,:])
# np.savetxt('b3.txt',b3_save[i,:])
# # np.savetxt('var.txt',var_save[i,:,:])

