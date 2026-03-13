# -*- coding: utf-8 -*-
"""
IRopex algorithm 
"""
import numpy as np
import time as tp 

class part1:

    def IRopex(n,K,L_F,L_H,variance_xi,variance_zeta,D_x, xi_test,xi_test_avg,zeta_test,zeta_test_avg,zeta_test_square,zeta,xi,x_star,x_init,tresh_eval,mu):
        theta_conv = 1
        tau_conv = 1
        tau_glob = 1
        eta = 1/pow(K,0.25)
        gamma = 1/(8*(L_F+ eta*L_H) + np.sqrt(K*(2*variance_xi + np.square(eta)*2*variance_zeta)))
        #eta = 100
        #gamma = 100
        
        x_previous = x_init.copy()
        x_current = x_init.copy()
        x_next = np.zeros(n)
        
        sum_tau = tau_glob
        sum_x_tau = tau_glob*x_current
     
        x_bar_curr = sum_x_tau/sum_tau
        x_bar_next = np.zeros(n)
        
        F_1_previous  = -2*x_previous[1] + xi[0]
        F_2_previous = 2*x_previous[0]
        
        F_1_current = F_1_previous 
        F_2_current  = F_2_previous
        
        H_1_previous  = x_previous[0] + zeta[0,0]
        H_2_previous = x_previous[1] + zeta[0,1]
        
        H_1_current = H_1_previous 
        H_2_current  = H_2_previous
         
        optim_gap = np.zeros(int((K+1)/tresh_eval)+1)
        feasib_gap = np.zeros(int((K+1)/tresh_eval)+1)
        iteration_time = np.zeros(int((K+1)/tresh_eval)+1)
        
        for k in range(K):
            if k == 0:
                optim_gap[k] = 0.5*(x_bar_curr @ x_bar_curr + 2* x_bar_curr @ zeta_test_avg + np.sum(zeta_test_square)) -  0.5*(x_star @ x_star + 2* x_star @ zeta_test_avg + np.sum(zeta_test_square))
                feasib_gap[k] = 25 -2*x_bar_curr[0]*x_star[1]+ xi_test_avg*x_bar_curr[0] - (25-2*x_star[0]*x_bar_curr[1]+ xi_test_avg *x_star[0])
                tStart_IRopex = tp.time()
            if (k+1) % tresh_eval == 0:
                iteration_time[int((k+1)/tresh_eval)] = tp.time() - tStart_IRopex
                optim_gap[int((k+1)/tresh_eval)] = 0.5*(x_bar_curr @ x_bar_curr + 2* x_bar_curr @ zeta_test_avg + np.sum(zeta_test_square)) -  0.5*(x_star @ x_star + 2* x_star @ zeta_test_avg + np.sum(zeta_test_square))
                feasib_gap[int((k+1)/tresh_eval)] = 25 -2*x_bar_curr[0]*x_star[1]+ xi_test_avg*x_bar_curr[0] - (25-2*x_star[0]*x_bar_curr[1]+ xi_test_avg *x_star[0])
                print(f"Iteration {k+1} of Ropex completed...")
                tStart_IRopex = tp.time()
            tau_strng_previous = k+1
            tau_strng_curr  = k+2 
            tau_glob = (mu==0)* tau_conv + (mu!=0) * tau_strng_curr
            theta_glob = (mu==0)* theta_conv + (mu!=0) *(tau_strng_curr/tau_strng_previous)
            
            ut_1 = F_1_current  + eta * H_1_current + theta_glob*(F_1_current + eta* H_1_current-(F_1_previous+eta*H_1_previous))
            ut_2 = F_2_current  + eta * H_2_current + theta_glob*(F_2_current + eta* H_2_current-(F_2_previous+eta*H_2_previous))
            
            x_next[0] = min(max(20,(x_current[0]-gamma*ut_1)),50)
            x_next[1] = min(max(5,(x_current[1]-gamma*ut_2)),15)
            
            x_current[0] = x_next[0]
            x_current[1] = x_next[1]
            
            F_1_previous = F_1_current
            F_2_previous = F_2_current
            H_1_previous = H_1_current
            H_2_previous = H_2_current
            
            F_1_current = -2*x_next[1] + xi[k]
            F_2_current = 2*x_next[0]
            H_1_current = x_next[0] + zeta[k,0]
            H_2_current = x_next[1] + zeta[k,1]
            
            sum_tau += tau_glob
            sum_x_tau += tau_glob*x_next
            
            x_bar_next = sum_x_tau/sum_tau
            x_bar_curr = x_bar_next
        return optim_gap, feasib_gap, iteration_time, x_bar_curr
        
        
        
            
            
        
        
    
    
    

