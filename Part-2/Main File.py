# -*- coding: utf-8 -*-
"""
Main file of second setting of minimization problem with traffic equilibrium problem 

The inner equilibrium problem  is a flow network containing 19 arcs 25 paths and 4 O-D
The full illustration of experiment is mentioned in the numerical experiment section of 
the paper. 
"""
#%% importing librarires 
import os
os.system('cls')
import numpy as np
import time
#import IROpex
from bilevel import part2
#import matplotlib as mtplot
import matplotlib.pyplot as plt
#import cvxpy as cp

%matplotlib qt 
"""
 The problem is defined based on the objective function psi(x)=E[0.5||x+zeta||^2]. 
 where x is a two-dimensional vector. the corresponding vector is 
 H  = [x_1 + zeta_1;x_2+zeta_2]. 
"""
tstart  = time.time()
R = 1 # total number of replications
K = 500 # total number of interations 
n = 8  # dimension of viariable x (path and O-D)
path_dim = 6 #number of paths
O_D_dim  = 2 
link_dim = 7
time_IRopex  = np.zeros(R)

time_IREG  = np.zeros(R)

tresh_eval = K/100 
optim_gap_IRopex  = np.zeros((R,int(K/tresh_eval)+1)) 
feasib_gap_IRopex  = np.zeros((R,int(K/tresh_eval)+1)) 
iteration_time_IRopex = np.zeros((R,int(K/tresh_eval)+1))

optim_gap_IREG  = np.zeros((R,int(K/tresh_eval)+1)) 
feasib_gap_IREG  = np.zeros((R,int(K/tresh_eval)+1)) 
iteration_time_IREG = np.zeros((R,int(K/tresh_eval)+1))
#%% Run each replication 
for r in range(R):
    #zeta = np.random.randn(2*K, O_D_dim)
    #xi = np.random.randn(2*K,O_D_dim)
    #xi_test = np.random.randn(min(5000000, 10*K),O_D_dim)
    #xi_test_avg = np.mean(xi_test,axis=0)
    #xi_test_square = np.mean(xi_test**2, axis=0)
    
    xi = np.zeros((2*K, O_D_dim))
    xi_test = np.zeros((min(5000000, 10*K), O_D_dim))

    xi_test_avg = np.mean(xi_test, axis=0)
    xi_test_square = np.mean(xi_test**2, axis=0)
    
    zeta = np.random.uniform(1, 1, size = (2*K,path_dim))
    zeta_test  = np.random.uniform(1, 1, size = (min(5000000, 10*K),path_dim))
    zeta_test_avg = np.mean(zeta_test,axis=0)
    
    n_a = 2*np.ones(link_dim) 
    t_a_0 = 1
    cap = 40*np.array([10,10,10,10,10,10,10],dtype=np.float64) #link capacity
    
    
    
    d = np.array([200,220],dtype = np.float64)
    
    Delta = np.array([[0,1,0,0,0,0], 
                     [0,0,0,1,0,1],
                     [1,1,0,1,1,0],
                     [0,0,1,0,0,1],
                     [0,0,0,0,1,0],
                     [1,0,1,0,0,0],
                     [1,0,0,1,0,0]], dtype = np.float64)
    
    Omega = np.array([[1,1,1,0,0,0],
                     [0,0,0,1,1,1]],dtype=np.float64)
    
    upper_path = np.array([200,250,350,400,500,700],dtype = np.float64)
    upper_O_D = Omega @ upper_path
    
    #var_obj = 1/3*np.linalg.norm(upper_path)
    
    #var_inner  = 3
    
    
    
    var_obj = 0
    
    var_inner  = 0
    
    L_H = 0 
    
    c_flow_deriv = t_a_0 * 0.15 * n_a / cap *  ((Delta @ upper_path) / cap)**(n_a-1)
    
    L_c = max(c_flow_deriv)
    
    L_F = np.sqrt(2*(np.linalg.norm(Delta.T)*L_c*np.linalg.norm(Delta))**2\
                  + np.linalg.norm(Omega)**2 + np.linalg.norm(Omega.T)**2)
    mu = 0 # set this to 1 for strongly montone case and to 0 for monotone case
    
    x_init = np.zeros(n)
    x_init[:path_dim] = np.random.uniform(0,upper_path)
    x_init[path_dim:] = np.random.uniform(0,upper_O_D)
    #%%
    tstart_IRopex = time.time()
    (
    optim_gap_IRopex[r, 0:int((K+1)/tresh_eval)+1],
    feasib_gap_IRopex[r, 0:int((K+1)/tresh_eval)+1],
    iteration_time_IRopex[r, 0:int((K+1)/tresh_eval)+1],
    x_bar_curr
    ) = part2.IRopex(
    n,path_dim,O_D_dim,link_dim,
    tresh_eval,
    xi,xi_test,xi_test_avg,
    zeta,zeta_test,zeta_test_avg,
    n_a,t_a_0,
    cap,d,Delta,Omega,upper_path,var_obj,var_inner,
    L_F,L_H,
    mu,K,x_init,upper_O_D
    
    )
    time_IRopex[r] = time.time()-tstart_IRopex
    
    tstart_IREG = time.time()
    (
    optim_gap_IREG[r, 0:int((K+1)/tresh_eval)+1],
    feasib_gap_IREG[r, 0:int((K+1)/tresh_eval)+1],
    iteration_time_IREG[r, 0:int((K+1)/tresh_eval)+1],
    y_bar_curr
    ) = part2.IREG(
    n,path_dim,O_D_dim,link_dim,
    tresh_eval,
    xi,xi_test,xi_test_avg,
    zeta,zeta_test,zeta_test_avg,
    n_a,t_a_0,
    cap,d,Delta,Omega,upper_path,var_obj,var_inner,
    L_F,L_H,
    mu,K,x_init,upper_O_D
    
    )
    time_IREG[r] = time.time()-tstart_IREG
    
    
    avg_iteration_time_IRopex  = np.mean(np.cumsum(iteration_time_IRopex,axis=1),axis=0)
    avg_opt_IRopex = np.mean(optim_gap_IRopex,axis=0)
    avg_feasib_IRopex = np.mean(feasib_gap_IRopex,axis=0)
    
    avg_iteration_time_IREG  = np.mean(np.cumsum(iteration_time_IREG,axis=1),axis=0)
    avg_opt_IREG = np.mean(optim_gap_IREG,axis=0)
    avg_feasib_IREG = np.mean(feasib_gap_IREG,axis=0)
    #%% Plot the results 
fig, ax = plt.subplots(2,2,figsize=(8,4))
title_font = {'fontsize': 14, 'fontweight': 'bold', 'fontname': 'serif'}
label_font = {'fontsize': 12, 'fontstyle': 'italic'}

ax[0,0].plot(tresh_eval*(np.arange(int(K/tresh_eval)) + 1),avg_opt_IRopex[1:],label = "Ropex",color = "blue", linewidth = 2)
ax[0,0].set_title("Optimality", fontdict=title_font)
ax[0,0].set_xlabel("Iteration", fontdict=label_font)
ax[0,0].set_ylabel(r"$\|\bar{x}_{k+1} - \bar{x}_{k}\|$")
ax[0,0].legend()
plt.tight_layout()
ax[0,1].plot(tresh_eval*(np.arange(int(K/tresh_eval)) + 1),avg_feasib_IRopex[1:],label = "Ropex",color = "blue", linewidth = 2)
ax[0,1].set_title("Feasibility", fontdict=title_font)
ax[0,1].set_xlabel("Iteration", fontdict=label_font)
ax[0,1].set_ylabel(r"$\phi(\bar{x}_k)$")
ax[0,1].legend()
plt.tight_layout()
ax[1,0].plot(avg_iteration_time_IRopex[1:],avg_opt_IRopex[1:],label = "Ropex",color = "blue", linewidth = 2)
ax[1,0].set_title("Optimality", fontdict=title_font)
ax[1,0].set_xlabel("Seconds", fontdict=label_font)
ax[1,0].set_ylabel(r"$\|\bar{x}_{k+1} - \bar{x}_{k}\|$")
ax[1,0].legend()
plt.tight_layout()
ax[1,1].plot(avg_iteration_time_IRopex[1:],avg_feasib_IRopex[1:],label = "Ropex",color = "blue", linewidth = 2)
ax[1,1].set_title("Feasiblity", fontdict=title_font)
ax[1,1].set_xlabel("Seconds", fontdict=label_font)
ax[1,1].set_ylabel(r"$\phi(\bar{x}_k)$")
ax[1,1].legend()
plt.tight_layout()
        
        
    
    
        
        
    
    
    
    
    

