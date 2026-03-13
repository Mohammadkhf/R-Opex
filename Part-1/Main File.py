# -*- coding: utf-8 -*-
"""
Part 1 of Experiments: We have a convex optimization problem with solution set
of a VI problem.  
"""

#%% importing librarires 
import os
os.system('cls')
import numpy as np
import time
#import ROpex
from bilevel import part1
#import matplotlib as mtplot
import matplotlib.pyplot as plt
#import cvxpy as cp

%matplotlib qt 
"""
 The problem is defined based on the objective function psi(x)=E[0.5||x+zeta||^2]. 
 where x is a two-dimensional vector. the corresponding operator is 
 H  = [x_1 + zeta_1;x_2+zeta_2]. 
"""
tstart  = time.time()
R = 10 # total number of replications
K = 5000000 # total number of interations 
n = 2 # dimension of viariable x
time_IRopex  = np.zeros(R)
tresh_eval = K/100 
optim_gap_IRopex  = np.zeros((R,int(K/tresh_eval)+1)) 
feasib_gap_IRopex  = np.zeros((R,int(K/tresh_eval)+1)) 
iteration_time_IRopex = np.zeros((R,int(K/tresh_eval)+1))


#%% Run each replication 
for r in range(R):
    zeta = np.random.randn(2*K, n)
    zeta_test = np.random.randn(min(5000000, 10*K),n)
    zeta_test_avg = np.mean(zeta_test,axis=0)
    zeta_test_square = np.mean(zeta_test**2, axis=0)
    
    xi = np.random.uniform(0, 20, size = 2*K)
    xi_test  = np.random.uniform(0, 20, size = min(5000000, 10*K))
    xi_test_avg = np.mean(xi_test)
    #xi_test_square = np.mean(xi_test**2)
    xi = np.random.normal(loc=10, scale=1, size = 2*K)
    xi_test  = np.random.normal(loc =10, scale =1, size = min(5000000, 10*K))
    xi_test_avg = np.mean(xi_test)
    
    x_star  = np.array([20,5],dtype=np.float64)
    
    x_init = np.array([25,10],dtype=np.float64)
    
    D_x  = 40 
    
    variance_zeta = 2
    variance_xi = pow(20,2)/123
    variance_zeta = 2
    variance_xi = 1
    
    l1 = 20
    u1 = 50 
    l2 = 5
    u2 = 15
    
    L_H = 1 
    F_decom = np.array([[0,-2],[2,0]])
    L_F = np.linalg.norm(F_decom)
    
    mu = 0 #change this to 1 for strongly-monotone case
    
    #%% Run the algorithm 
    tstart_IRopex = time.time()
    optim_gap_IRopex[r,0:int((K+1)/tresh_eval)+1], feasib_gap_IRopex[r,0:int((K+1)/tresh_eval)+1], iteration_time_IRopex[r,0:int((K+1)/tresh_eval)+1], x_bar_curr = part1.IRopex(n,K,L_F,L_H,variance_xi,variance_zeta,D_x, xi_test,xi_test_avg,zeta_test,zeta_test_avg,zeta_test_square,zeta,xi,x_star,x_init,tresh_eval,mu)
    time_IRopex[r] = time.time()-tstart_IRopex
    avg_iteration_time_IRopex  = np.mean(np.cumsum(iteration_time_IRopex,axis=1),axis=0)
    avg_opt_IRopex = np.mean(optim_gap_IRopex,axis=0)
    avg_feasib_IRopex = np.mean(feasib_gap_IRopex,axis=0)
    
#%% Plot the results 
fig, ax = plt.subplots(2,2,figsize=(8,4))
title_font = {'fontsize': 14, 'fontweight': 'bold', 'fontname': 'serif'}
label_font = {'fontsize': 12, 'fontstyle': 'italic'}

ax[0,0].plot(tresh_eval*(np.arange(int(K/tresh_eval)) + 1),avg_opt_IRopex[1:],label = "Ropex",color = "blue", linewidth = 2)
ax[0,0].set_title("Optimality", fontdict=title_font)
ax[0,0].set_xlabel("Iteration", fontdict=label_font)
ax[0,0].set_ylabel(r"$\psi(\bar{x}_k) - \psi(x^\ast)$")
ax[0,0].legend()
plt.tight_layout()
ax[0,1].plot(tresh_eval*(np.arange(int(K/tresh_eval)) + 1),avg_feasib_IRopex[1:],label = "Ropex",color = "blue", linewidth = 2)
ax[0,1].set_title("Feasibility", fontdict=title_font)
ax[0,1].set_xlabel("Iteration", fontdict=label_font)
ax[0,1].set_ylabel(r"$f(\bar{x}_{1,k}, x^*_2) - f( x^*_1, \bar{x}_{2,k})$")
ax[0,1].legend()
plt.tight_layout()
ax[1,0].plot(avg_iteration_time_IRopex[1:],avg_opt_IRopex[1:],label = "Ropex",color = "blue", linewidth = 2)
ax[1,0].set_title("Optimality", fontdict=title_font)
ax[1,0].set_xlabel("Seconds", fontdict=label_font)
ax[1,0].set_ylabel(r"$\psi(\bar{x}_k) - \psi(x^\ast)$")
ax[1,0].legend()
plt.tight_layout()
ax[1,1].plot(avg_iteration_time_IRopex[1:],avg_feasib_IRopex[1:],label = "Ropex",color = "blue", linewidth = 2)
ax[1,1].set_title("Feasiblity", fontdict=title_font)
ax[1,1].set_xlabel("Seconds", fontdict=label_font)
ax[1,1].set_ylabel(r"$f(\bar{x}_{1,k}, x^*_2) - f( x^*_1, \bar{x}_{2,k})$")
ax[1,1].legend()
plt.tight_layout()
    
    
                            
    