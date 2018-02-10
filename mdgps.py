
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
import GPy
from termcolor import colored
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))


# In[ ]:

from model import unicycle
from cost import tracking
from iLQR import iLQR
from poliOpt import poliOpt
from appPoliOpt import appPoliOpt
from trajOpt import trajOpt
from policyClass import localPolicy
from modelLearn import localModelLearn, localPolicyLearn
from gps_util import getCostNN, getCostAppNN, getCostTraj, getObs, getDesired, getPlot

class mdGPS :
    def __init__(self,name):
        # class name
        self.name = name
        
        # size of state, input, horizon, hidden node
        self.ix = 3
        self.iu = 2
        self.io = 3
        self.N = 100
        self.hidden_num = 40
        
                
        # the number of initial position 
        self.num_ini = 2
        self.x0 = np.array( [ [-2.0,-0.5,np.pi/2],[2.0,-0.5,np.pi/2] ] )
        # self.x0 = np.expand_dims( np.array( [-2.0,-0.5,np.pi/2] ), axis=0 )
        # self.x0 = np.array( [ [0,-2,np.pi/2]] )
        self.u0 = np.zeros( (self.N,2) )
        self.x0_pre = np.expand_dims( np.array( [0.0,-0.5,np.pi/2] ), axis=0 )
        self.x_t = np.array([0,3])
        # self.stepIni = 0.005
        self.stepIni = 0.005

        # flag for state & input constraints
        self.flag_const = False
        
        # policy, traj class, real-robot model
        self.myModel = unicycle('unicycle')
        self.myCost = tracking('unicycle',self.x_t,self.N,self.flag_const)
        self.myPolicy = poliOpt('unicycle',self.hidden_num,5000,self.stepIni,self.io,self.iu,self.N)
        self.myTraj = trajOpt('unicycle',self.N,self.myModel,self.myCost,2)
        self.myAppPolicy = appPoliOpt('unicycle',20,100,self.stepIni,self.ix,self.iu,self.N,True)
        
        # step size used in step adjustment
        self.epsilon = 5
        self.eps_max = 1e16
        self.eps_min = 1e-6
        
        # maxIter used for update function
        self.maxIter = 20
        
        # the number of sample for fitted model, which determines how many times robot actually works
        self.num_fit = 10
        
        # the number of sample for policy optimization / samples are drawn from the local policy from iLQR
        self.num_sample = self.num_fit
        
        # NN iter
        self.maxIterNN = 50
        
        # policy type
        self.onPolicy = True
        
        # is state different with obsevation?
        self.obsFlag = False

        # use kinematic model?
        self.flagKinModel = True

        # do you want use NN ver appoximated global policy?
        self.flagAppNN = False

    def update(self) :
        # main loop for mdgps

        # time horizon & dimension
        N = self.N
        ix = self.ix
        iu = self.iu
        io = self.io
        
        # saving NN parameters // not used currently
        W1_save = np.zeros((self.maxIter+1,self.ix,self.hidden_num))
        W2_save = np.zeros((self.maxIter+1,self.hidden_num,self.hidden_num))
        W3_save = np.zeros((self.maxIter+1,self.hidden_num,self.iu))
        b1_save = np.zeros((self.maxIter+1,self.hidden_num))
        b2_save = np.zeros((self.maxIter+1,self.hidden_num))
        b3_save = np.zeros((self.maxIter+1,self.iu))
        var_save = np.zeros((self.maxIter+1,self.N,self.iu,self.iu))
        
        # stepsize setting for supervised learning (policy optimization)
        stepIni = self.stepIni
        stepSize = stepIni
           
        # dual variable for kl divergence inequality constraint
        eta_ini = 1
        eta = eta_ini
        
        # initial cost / set inf
        cNmP = 1e8
        cNmN = 1e8
        cNmN_pre = 1e8
        cPmP_NN = 1e8 
        cNmN_NN = 1e8
        
        # real cost from robot
        cost_real = np.zeros(self.maxIter)
        
        # Initial policy from iLQR under known nominal model // pre-training for NN
        W1,b1,W2,b2,W3,b3,pol_var,pol_var_inv,x_nominal_ini,u_nominal_ini, iniPolicy = self.getInitialPolicy(stepIni)
        
        # saving initial NN parameters
        W1_save[0,:,:] = W1
        W2_save[0,:,:] = W2
        W3_save[0,:,:] = W3
        b1_save[0,:] = b1
        b2_save[0,:] = b2
        b3_save[0,:] = b3
        var_save[0,:,:,:] = pol_var
        
        # initialize local policy, local approximated global policy
        localPolicySet = [''] * self.num_ini
        appGlobalPolicySet = [''] * self.num_ini
        
        # initialize fitted model & approximated global policy
        myFitModelSet = [''] * self.num_ini
        myFitPolicySet = [''] * self.num_ini
        for j in range(self.num_ini) :
            myFitModelSet[j] = localModelLearn('hi',self.ix,self.iu,self.N,10,np.ones((N,8,1)))
            myFitPolicySet[j] = localPolicyLearn('hi',self.ix,self.iu,self.N,10,np.ones((N,5,1)))
        myFitModelOldSet = myFitModelSet
        
        ####################### Variables ###########################
        # x_save ---> data for supervised learning of policy
        # x_new, x_traj  ---> the result of trajectory optimization along initial position
        # x_fit ---> data to fitting linearized policy
        # x_ini  ---> data for initial policy opitmization, it is resutled from iLQR.
        # space for trajectory from optimization along initial condition
        x_traj = np.zeros((self.N+1,self.ix,self.num_ini))
        u_traj = np.zeros((self.N,self.iu,self.num_ini))
        # K_traj = np.zeros((self.num_ini,self.N,self.iu,self.ix))
        # Quu_inv_traj = np.zeros((self.num_ini,self.N,self.iu,self.iu))
        
        # flag
        print("======================== Mirror descent GPS ============================ ")
        for i in range(self.maxIter) :
            print colored('iteration = ', 'blue'), colored(i+1, 'blue')
            print("trajectory optimization start!!")
            
            if i > 0 :
                # self.num_fit = 10
                self.num_sample = self.num_fit

            # sample data
            x_fit = [''] * self.num_ini
            u_fit_m = [''] * self.num_ini
            o_fit = [''] * self.num_ini
            xNominal = [''] * self.num_ini
            uNominal = [''] * self.num_ini 

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
                print(j, "initial position for sample generation")
                
                # 1. move real robot to get sample used for model learning, approximate global policy
                print("move the robot !!")
                if i == 0:
                    xNominal[j], uNominal[j] = x_nominal_ini, u_nominal_ini
                    x_fit[j], u_fit_m[j], cost_fit, o_fit[j] = self.driveRobot(self.x0[j,:],self.onPolicy,i,iniPolicy,False) 
                else :
                    xNominal[j], uNominal[j] = localPolicySet[j].x_nominal,localPolicySet[j].u_nominal
                    x_fit[j], u_fit_m[j], cost_fit, o_fit[j] = self.driveRobot(self.x0[j,:],self.onPolicy,i,localPolicySet[j],False)
                cost_real_ini[j] = np.mean(cost_fit)             
                
                # xNominal is changed along to the current j'th iteration
                
                # 2. fit linear gaussian global dynamics
                myFitModelOldSet[j] = myFitModelSet[j]
                if self.flagKinModel == False :
                    print("fit the local model !!")
                    myFitModelSet[j].update(x_fit[j],u_fit_m[j])
                    if i == 0 : 
                        myFitModelSet[j].data_aug(x_fit[j],u_fit_m[j],True)
                    else : 
                        myFitModelSet[j].data_aug(x_fit[j],u_fit_m[j])
                else :
                    # sample data was not used
                    tempA, tempB = self.myModel.diffDyn(xNominal[j][0:N,:],uNominal[j])
                    tempC = self.myModel.forwardDyn(xNominal[j][0:N,:],uNominal[j]) \
                             - np.squeeze(np.matmul(tempA,np.expand_dims(xNominal[j][0:N,:],axis=2))) \
                             - np.squeeze(np.matmul(tempB,np.expand_dims(uNominal[j],axis=2)))
                    myFitModelSet[j].setter(tempA,tempB,tempC)

            # fitting approx policy using neural net
            if self.flagAppNN == True :
                print("NN or GP ver :fit the approximated global model !!")
                self.myAppPolicy.setEnv(100,0.001)
                appGlobalPolicySet2 = self.fittingApproxPolicy2(x_fit,o_fit,xNominal,uNominal,myFitModelSet,self.x0)
            else :
                pass
                    
            for j in range(self.num_ini) :

                print(j, "initial position in trajectory optimization")
                if self.flagAppNN == False :
                    # 3. fit approximated global policy using samples
                    print("fit the approximated global model !!")
                    appGlobalPolicySet[j], myFitPolicySet[j] = self.fittingApproxPolicy(x_fit[j],o_fit[j],myFitPolicySet[j])
                    # K_fit = appGlobalPolicySet[j].K_mat
                    # k_fit = appGlobalPolicySet[j].k_mat
                    if i == 0 :
                        myFitPolicySet[j].data_aug(x_fit[j],appGlobalPolicySet[j].u_nominal,True)
                    else : 
                        myFitPolicySet[j].data_aug(x_fit[j],appGlobalPolicySet[j].u_nominal)
                else :
                    pass

                # print appGlobalPolicySet2[j].K_mat[0:4,:,:]
                # print appGlobalPolicySet[j].K_mat[0:4,:,:]
                # appGlobalPolicySet[j] = appGlobalPolicySet2[j]
               
                # 4. trajectory optimization 
                self.myTraj.setEnv(self.myPolicy,eta,self.epsilon,
                                    appGlobalPolicySet[j].K_mat,appGlobalPolicySet[j].k_mat,myFitModelSet[j],self.myAppPolicy,False)
                x_new,u_new, Quu_new, Quu_inv_new, eta_new, cost_new, K_new, k_new = self.myTraj.iterDGD(self.x0[j,:],uNominal[j])
                localPolicySet[j] = localPolicy("local",ix,iu,N)
                localPolicySet[j].setter(K_new,k_new,x_new,u_new,Quu_inv_new)
                x_traj[:,:,j] = x_new
                u_traj[:,:,j] = u_new
                
                # cost, eta save
                eta_save[j] = eta_new
                cost_save_cm[j] = getCostTraj(self.x0[j,:],localPolicySet[j],N,self.myTraj,myFitModelSet[j])
                cost_save_pm[j] = getCostTraj(self.x0[j,:],localPolicySet[j],N,self.myTraj,myFitModelOldSet[j])
                # new setting
                eta = eta_new

                # 5. generate samples for supervised learning
                num_sample = self.num_sample
                for k in range(num_sample):
                    # sample extraction
                    x_sample = x_fit[j][:,:,k]
                    u_sample = np.zeros((N,iu))
                    u_sample = u_new + k_new + np.squeeze( np.matmul(K_new,np.expand_dims(x_sample[0:N,:]-x_new[0:N,:],axis=2)))
                    o_sample = o_fit[j][:,:,k]

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
            # getPlot(np.expand_dims(x_new,axis=2),np.expand_dims(u_new,axis=2),self.x_t,1,N)    

            # 6. policy optimzation using supervised learning based on SGD
            print("policy optimization start!!")
            stepSize = stepSize * 0.95
            self.myPolicy.setEnv(self.maxIterNN,stepIni)
            for jj in range(20) :               
                W1,b1,W2,b2,W3,b3,pol_var,pol_var_inv = self.myPolicy.update(o_save,u_save,var_inv_save * np.mean(eta_save),
                                                                            var_inv_set_save,self.num_ini*self.num_sample)
                c_temp = np.zeros(self.num_ini)
                for j in range(self.num_ini) :
                    c_temp[j] = getCostAppNN(self.x0[j,:],appGlobalPolicySet[j],N,self.myTraj,myFitModelSet[j])
                    
                if np.mean(c_temp) >= 20 * cNmN or np.isnan(np.mean(c_temp)) :
                    print colored('policy optimization is diversed cost is', 'red'), colored(np.mean(c_temp), 'red')
                    # flag_eps = False
                    stepSize = stepIni
                    self.myPolicy.setEnv(self.maxIterNN, stepSize / (10*jj+1))
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
            # self.testRobot(self.x0,x_new,u_new) 
            self.driveRobot(self.x0,True,i,localPolicySet[0],True,x_traj,u_traj)
            # driveRobot(self,x0,onPolicy,mainIter,localPolicy,flag_test,x_new=None,u_new=None)
            # 7. terminal condition (when the change of cNmN is extremly low)            
            if np.abs(cNmN_NN - cPmP_NN) < 0.01 :
                print("SUCCEESS : cost change < tolFun")
                break
            else :
                cNmN_pre = cNmN
            
                
        return W1_save, W2_save, W3_save, b1_save, b2_save, b3_save, var_save,i,cost_real
              
                
    def fittingApproxPolicy(self,x_fit,o_fit,myFitPolicy) :
        # variables
        N = self.N
        ix = self.ix
        iu = self.iu
        io = self.io
        num_fit = self.num_fit
        
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
   
        tempPolicy = localPolicy("global",ix,iu,N)
        tempPolicy.setter(myFitPolicy.K,myFitPolicy.k,x_fit,u_fit_p,var_temp)
            
        return tempPolicy, myFitPolicy

    def fittingApproxPolicy2(self,x_fit,o_fit,xNominal,uNominal,myFitModelSet,x0) :
        # variables
        N = self.N
        ix = self.ix
        iu = self.iu
        io = self.io
        num_fit = self.num_fit
        num_ini = self.num_ini
        
        K_fit = np.zeros((N,iu,ix))
        k_fit = np.zeros((N,iu))
        
        u_fit_p = np.zeros((N,iu,num_fit,num_ini))
        x_temp = np.zeros((N,ix,num_fit,num_ini)) 
        x_nominal = np.zeros((N+1,ix,num_ini))
        
        # data augmentation
        for i in range(num_ini) :
            for ip in range(N) :
                for im in range(num_fit) :
                    # sample from global policy
                    o_temp = o_fit[i][ip,:,im]     
                    u_fit_p[ip,:,im,i], var_temp = self.myPolicy.getPolicy(o_temp)
                    x_temp[ip,:,im,i] = x_fit[i][ip,:,im] 
        u_fit_p = np.reshape(u_fit_p,(N*num_fit*num_ini,iu))
        x_temp = np.reshape(x_temp,(N*num_fit*num_ini,ix))


        # learning policy
        self.myAppPolicy.update(x_temp, u_fit_p,num_fit*num_ini)

        # nominal x setting
        for i in range(num_ini) :
            for ip in range(N+1) :
                if ip == 0 :
                    x_nominal[ip,:,i] = x0[i,:]
                else :
                    x_nominal[ip,:,i] = myFitModelSet[i].forwardDyn(x_nominal[ip-1,:,i],uNominal[i][ip-1,:],ip-1)
                # x_nominal[:,:,i] = xNominal[i]
                # x_nominal[:,:,i] = np.squeeze( np.mean(x_fit[i][0:N,:,:],axis=2) )
                # x_nominal[:,:,i] = np.zeros((N,ix))

        # get jacobian
        tempPolicy = [''] * num_ini    
        for i in range(num_ini) :
            tempPolicy[i] = localPolicy("global",ix,iu,N)
            K_fit = self.myAppPolicy.jacobian(x_nominal[0:N,:,i])
            k_fit = self.myAppPolicy.getPolicy(x_nominal[0:N,:,i]) \
                             - np.squeeze( np.matmul(K_fit,np.expand_dims(x_nominal[0:N,:,i],axis=2)) )
            tempPolicy[i].setter(K_fit,k_fit,x_nominal[:,:,i],uNominal[i])
            
        return tempPolicy
                                   
    def driveRobot(self,x0,onPolicy,mainIter,localPolicy,flag_test,x_new=None,u_new=None) :
        
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
        if flag_test == True :
            num_fit = self.num_ini
        else :
            num_fit = self.num_fit
        
        # data used for model learning
        x_fit = np.zeros((N+1,ix,num_fit))
        o_fit = np.zeros_like(x_fit)
        u_fit_m = np.zeros((N,iu,num_fit))
        cost_fit = np.zeros(num_fit)

        for im in range(num_fit):
            if flag_test == True :
                x_fit[0,:,im] = x0[im,:]
            else :
                x_fit[0,:,im] = x0 + np.array((1,1,0.3)) * (np.random.random(3) - 0.5) * 0.2
            o_fit[0,:,im] = getObs(self.x_t,x_fit[0,:,im],self.obsFlag)

            for ip in range(N) :
                if onPolicy == False:       
                    u_temp = np.random.multivariate_normal(u_traj[ip,:] + k_traj[ip,:] + np.dot(K_traj[ip,:,:], \
                                                           x_fit[ip,:,im] - x_traj[ip,:]),Quu_inv_traj[ip,:,:] / 1)             
                else :                   
                    o_temp = o_fit[ip,:,im]     
                    u_temp, var_temp = self.myPolicy.getPolicy(o_temp)
                    if flag_test == False :
                        u_temp = np.random.multivariate_normal(np.squeeze(u_temp), var_temp[ip,:,:] / 1 )
                    else :
                        pass
                    u_temp = np.squeeze(u_temp)               
                u_fit_m[ip,:,im] = u_temp
                x_fit[ip+1,:,im] = self.myModel.forwardDyn(x_fit[ip,:,im], u_fit_m[ip,:,im], ip)
                o_fit[ip+1,:,im] = getObs(self.x_t,x_fit[ip+1,:,im],self.obsFlag)

            cost_fit[im] = self.myTraj.getCost(x_fit[:,:,im],u_fit_m[:,:,im])
            print colored('mean value of real cost is', 'yellow'), colored(cost_fit[im], 'yellow')
        if flag_test == True :
            getPlot(x_fit,u_fit_m,self.x_t,num_fit,N,x_new,u_new)
            return None
        else :
            getPlot(x_fit,u_fit_m,self.x_t,num_fit,N)  
            return x_fit, u_fit_m, cost_fit, o_fit
                
    def getInitialPolicy(self,stepIni) :
        
        print("initial trajectories from iLQR")  
        # Initial trajectory distribution from iLQR with low number of iterations
        i1 = iLQR('unicycle',self.N,2,self.myModel,self.myCost)
        x0 = self.x0_pre
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

        # iLQR with prior model without constraints
        x, u, Quu, Quu_inv, K, k_mat = i1.update(x0[0,:],u0)
        u = u 
        K = K 
        k_mat = k_mat 

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
        
        iniPolicy = localPolicy("local",ix,iu,N)
        
        print("initial policy optimization starts!!")
        self.myPolicy.setEnv(self.maxIterNN,stepIni)
        W1,b1,W2,b2,W3,b3,pol_var,pol_var_inv = self.myPolicy.update(o_ini,u_ini,var_inv_ini,var_inv_set_ini,num_sample)
        # costNN_pre = getCostNN(x0[0,:],self.myPolicy,N,self.myTraj,self.myModel)
        # print colored('initial policy parameter is updated true cost is ', 'green'), colored(costNN_pre, 'green')
        iniPolicy.setter(K, k_mat, x, u, pol_var)
        
        return W1,b1,W2,b2,W3,b3,pol_var,pol_var_inv,x,u, iniPolicy


