
# coding: utf-8

# In[ ]:

from __future__ import division
import numpy as np

def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))
# cost calculation function
def getCostNN(x0,policy,N,traj,model):
    x_traj = np.zeros((N+1,traj.ix))
    u_traj = np.zeros((N,traj.iu))
    x_traj[0,:] = x0
    for i in range(N) :
        if i == 0 :
            x_temp = np.vstack((x_traj[i,:],x_traj[i,:]))
        else : 
            x_temp = np.vstack((x_traj[i-1,:],x_traj[i,:]))
          
        u_temp, var_temp = policy.getPolicy(x_temp)
        # u_temp = np.random.multivariate_normal(np.squeeze(u_temp), var_temp[i,:,:] )
        u_traj[i,:] = u_temp
        x_traj[i+1,:] = model.forwardDyn(x_traj[i,:],u_traj[i,:],i)     
    return traj.getCost(x_traj,u_traj)

def getCostAppNN(x0,policy,N,traj,model):
#     x = policy.x_nominal
    k = policy.k_mat
    K = policy.K_mat
    Quu_inv = policy.polVar
    
    x_traj = np.zeros((N+1,traj.ix))
    u_traj = np.zeros((N,traj.iu))
    x_traj[0,:] = x0
    for i in range(N) :
        x_temp = x_traj[i,:]
#         x_temp = np.expand_dims(x_traj[i,:],axis=0)
        u_temp = np.dot(K[i,:,:],x_temp) + k[i,:]
        u_traj[i,:] = u_temp
        x_traj[i+1,:] = model.forwardDyn(x_traj[i,:],u_traj[i,:],i)     
    return traj.getCost(x_traj,u_traj)

# (self.x0,localPolicy,N,self.myTraj,self.myFitModel)
def getCostTraj(x0,policy,N,traj,model): 
    x = policy.x_nominal
    u = policy.u_nominal
    k = policy.k_mat
    K = policy.K_mat
    Quu_inv = policy.polVar
    
    x_traj = np.zeros((N+1,traj.ix))
    u_traj = np.zeros((N,traj.iu))
    x_traj[0,:] = x0
    for i in range(N) :
        u_temp = u[i,:] + k[i,:] + np.dot(K[i,:,:],x_traj[i,:] - x[i,:])
        u_traj[i,:] = u_temp
        x_traj[i+1,:] = model.forwardDyn(x_traj[i,:],u_traj[i,:],i)
    return traj.getCost(x_traj,u_traj)

def getObs(P_t,P_c,Flag) :
    
    # desired position
    x_d = P_t[0]
    y_d = P_t[1]
    
    # robot position
    x = P_c[0]
    y = P_c[1]
    yaw = P_c[2]
    
    # rotation matrix
    R = np.array( [ [np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]] )
    trans = np.expand_dims(np.array([x,y]),-1)
    
    # Homogeneous transform
    H = np.vstack((np.hstack([R.T,np.dot(-R.T,trans)]),np.array([0,0,1])))
    
    # translation
    P_T = np.zeros(3)
    P_T[0:2] = P_t[0:2]
    P_T[2] = 1
    
    P_t_local = np.dot(H,P_T)
    yaw_diff = np.arctan2(P_t_local[1],P_t_local[0])
    
    if Flag == True :
        output = np.hstack((P_t_local[0:2],yaw_diff))
    else :
        output = P_c
    
    return output
    
def getDesired(P_t,P_c) :
    
    # distance
    dist = np.linalg.norm(P_t-P_c)
    # desired x,y position
    P_des = P_t - (P_t - P_c) / dist
    # desired yaw
    P_diff = P_t - P_c
    yaw_des = np.arctan2(P_diff[1],P_diff[0])

    output = np.hstack((P_des,yaw_des))

    return output