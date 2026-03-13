# -*- coding: utf-8 -*-
"""
IRopex algorithm : second setting
"""
import numpy as np
import time as tp 


def F_operator(n,x,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi):
    c_delta_path = t_a_0*(1+0.15*((Delta @ x[:path_dim])/cap)**n_a)
    C_path = Delta.T @ c_delta_path
    F = np.zeros(n)
    F[:path_dim] = C_path - Omega.T @ x[path_dim:]
    F[path_dim:] = Omega @ x[:path_dim] - (d + xi)
    return F
 
def H_operator(n,x,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,zeta):
    c_flow_derivative = t_a_0 * 0.15 * n_a / cap *  ((Delta @ x[:path_dim]) / cap)**(n_a-1)
    grad_H_h = (Delta @ zeta).T @ np.diag(c_flow_derivative) @ Delta
    H = np.zeros(n)
    H[:path_dim] = grad_H_h
    H[path_dim:] = 0
    return H
    
class part2:
    def IREG(n, path_dim, O_D_dim, link_dim,
    tresh_eval,
    xi, xi_test, xi_test_avg,
    zeta, zeta_test, zeta_test_avg,
    n_a, t_a_0,
    cap, d, Delta, Omega, upper_path, var_obj, var_inner,
    L_F, L_H,
    mu, K, x_init, upper_O_D):
        #y_cur =  x_init.copy()
        x_cur =  x_init.copy()
        y_bar_curr = x_init.copy() 
        
        y_next = x_init.copy()
        x_next = x_init.copy()
        y_bar_next = x_init.copy() 
        
        y_bar_prev = y_bar_curr
        
        eta = 1/pow(1,0.5) 
        
        eta_cur = eta 
        
        F_full = F_operator(n,x_cur,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi[0,:]) 
        F_1_current = F_full[:path_dim]
        F_2_current =  F_full[path_dim:] 
        
        
        
        
        
        H_full = H_operator(n,x_cur,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,zeta[0,:])
        H_1_current =  H_full[:path_dim]
        H_2_current = H_full[path_dim:]
        
        
        
        optim_gap = np.zeros(int((K+1)/tresh_eval)+1)
        feasib_gap = np.zeros(int((K+1)/tresh_eval)+1)
        iteration_time = np.zeros(int((K+1)/tresh_eval)+1)
        
        for k in range(K):
            
            eta_next = eta/pow((k+1),0.5)
            gamma = np.sqrt(0.5/(L_F**2+ (eta*L_H)**2))
        
            if k == 0:
                optim_gap[k] = np.linalg.norm(y_bar_curr - y_bar_prev)
                feasib_gap[k] = np.linalg.norm(np.maximum(0,-y_bar_curr))**2\
                    +np.linalg.norm(np.maximum(0,-F_operator(n,y_bar_curr,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi_test_avg)))**2\
                        +np.linalg.norm(y_bar_curr @ F_operator(n,y_bar_curr,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi_test_avg))
                tStart_IREG = tp.time()
            if (k+1) % tresh_eval == 0:
                iteration_time[int((k+1)/tresh_eval)] = tp.time() - tStart_IREG
                optim_gap[int((k+1)/tresh_eval)] = np.linalg.norm(y_bar_curr - y_bar_prev)
                feasib_gap[int((k+1)/tresh_eval)] = np.linalg.norm(np.maximum(0,-y_bar_curr))**2\
                    +np.linalg.norm(np.maximum(0,-F_operator(n,y_bar_curr,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi_test_avg)))**2\
                        +np.linalg.norm(y_bar_curr @ F_operator(n,y_bar_curr,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi_test_avg))
                print(f"Iteration {k+1} of IE_EG completed...")
                tStart_IREG = tp.time()
            
            F_full_cur = F_operator(n,x_cur ,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi[k,:])
            F_1_current = F_full_cur[:path_dim]
            F_2_current = F_full_cur[path_dim:]
            
            H_full_cur = H_operator(n,x_cur,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,zeta[0,:])
            H_1_current = H_full_cur[:path_dim]
            H_2_current = H_full_cur[path_dim:]
            
            ut_path_y = F_1_current  + eta_cur * H_1_current
            ut_O_D_y = F_2_current  + eta_cur * H_2_current 
            
            y_next[:path_dim] = np.minimum(np.maximum(0,(x_cur[:path_dim]-gamma*ut_path_y)),upper_path)
            y_next[path_dim:] = np.minimum(np.maximum(0,(x_cur[path_dim:]-gamma*ut_O_D_y)),upper_O_D)
            
            F_full_next_x = F_operator(n,y_next,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi[k,:])
            H_full_next_x = H_operator(n,y_next,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,zeta[0,:])
           
            F_1_current_x =  F_full_next_x[:path_dim]
            F_2_current_x =  F_full_next_x[path_dim:]
           
            H_1_current_x =  H_full_next_x[:path_dim]
            H_2_current_x =  H_full_next_x[path_dim:]
           
           
           
            ut_path_x = F_1_current_x  + eta_cur * H_1_current_x
            ut_O_D_x = F_2_current_x  + eta_cur * H_2_current_x
            
            x_next[:path_dim] = np.minimum(np.maximum(0,(x_cur[:path_dim]-gamma*ut_path_x)),upper_path)
            x_next[path_dim:] = np.minimum(np.maximum(0,(x_cur[path_dim:]-gamma*ut_O_D_x)),upper_O_D)
            
            x_cur = x_next
            y_bar_next = (k * y_bar_curr + y_next)/(k+1)
            y_bar_prev = y_bar_curr
            y_bar_curr = y_bar_next
            
            eta_cur = eta_next
        return optim_gap, feasib_gap, iteration_time, y_bar_curr
        
            
    def IRopex( n, path_dim, O_D_dim, link_dim,
     tresh_eval,
     xi, xi_test, xi_test_avg,
     zeta, zeta_test, zeta_test_avg,
     n_a, t_a_0,
     cap, d, Delta, Omega, upper_path, var_obj, var_inner,
     L_F, L_H,
     mu, K, x_init, upper_O_D):
        theta_conv = 1
        tau_conv = 1
        tau_glob = 1
        #eta = 1/pow(K,0.25)
        eta = 1/pow(1,0.5)
        #gamma = 1/(8*(L_F+ eta*L_H) + np.sqrt(K*(2*var_inner + np.square(eta)*2*var_obj)))
        
        gamma = np.sqrt(0.5/(L_F**2+ (eta*L_H)**2))
        
        gamma = 1/(100 * eta+ 10*np.sqrt(K))
        
        
        x_previous = x_init.copy()
        x_current = x_init.copy()
        x_next = np.zeros(n)
        
        sum_tau = tau_glob
        sum_x_tau = tau_glob*x_current
        
        x_bar_curr = sum_x_tau/sum_tau
        x_bar_next = np.zeros(n)
        x_bar_prev = x_bar_curr
        
        F_full = F_operator(n,x_previous,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi[0,:])
        
        F_1_previous  = F_full[:path_dim]
        F_2_previous = F_full[path_dim:]
        
        F_1_current = F_1_previous 
        F_2_current  = F_2_previous
        
        H_full = H_operator(n,x_previous,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,zeta[0,:])
        
        H_1_previous  = H_full[:path_dim]
        H_2_previous = H_full[path_dim:]
        
        H_1_current = H_1_previous 
        H_2_current  = H_2_previous
         
        optim_gap = np.zeros(int((K+1)/tresh_eval)+1)
        feasib_gap = np.zeros(int((K+1)/tresh_eval)+1)
        iteration_time = np.zeros(int((K+1)/tresh_eval)+1)
        
        for k in range(K):
            eta = 1/pow((k+1),0.5)
            gamma = np.sqrt(0.5/(L_F**2+ (eta*L_H)**2))
        
            if k == 0:
                optim_gap[k] = np.linalg.norm(x_bar_curr - x_bar_prev)
                feasib_gap[k] = np.linalg.norm(np.maximum(0,-x_bar_curr))**2\
                    +np.linalg.norm(np.maximum(0,-F_operator(n,x_bar_curr,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi_test_avg)))**2\
                        +np.linalg.norm(x_bar_curr @ F_operator(n,x_bar_curr,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi_test_avg))
                tStart_IRopex = tp.time()
            if (k+1) % tresh_eval == 0:
                iteration_time[int((k+1)/tresh_eval)] = tp.time() - tStart_IRopex
                optim_gap[int((k+1)/tresh_eval)] = np.linalg.norm(x_bar_curr - x_bar_prev)
                feasib_gap[int((k+1)/tresh_eval)] = np.linalg.norm(np.maximum(0,-x_bar_curr))**2\
                    +np.linalg.norm(np.maximum(0,-F_operator(n,x_bar_curr,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi_test_avg)))**2\
                        +np.linalg.norm(x_bar_curr @ F_operator(n,x_bar_curr,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi_test_avg))
                print(f"Iteration {k+1} of Ropex completed...")
                tStart_IRopex = tp.time()
            tau_strng_previous = k+1
            tau_strng_curr  = k+2 
            tau_glob = (mu==0)* tau_conv + (mu!=0) * tau_strng_curr
            theta_glob = (mu==0)* theta_conv + (mu!=0) *(tau_strng_curr/tau_strng_previous)
            
            ut_path = F_1_current  + eta * H_1_current + theta_glob*(F_1_current + eta* H_1_current-(F_1_previous+eta*H_1_previous))
            ut_O_D = F_2_current  + eta * H_2_current + theta_glob*(F_2_current + eta* H_2_current-(F_2_previous+eta*H_2_previous))
            
            x_next[:path_dim] = np.minimum(np.maximum(0,(x_current[:path_dim]-gamma*ut_path)),upper_path)
            x_next[path_dim:] = np.minimum(np.maximum(0,(x_current[path_dim:]-gamma*ut_O_D)),upper_O_D)
            
            x_current[:path_dim] = x_next[:path_dim]
            x_current[path_dim:] = x_next[path_dim:]
            
            
            
            F_1_previous = F_1_current
            F_2_previous = F_2_current
            H_1_previous = H_1_current
            H_2_previous = H_2_current
            
            F_full_next = F_operator(n,x_next,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,xi[k,:])
            H_full_next = H_operator(n,x_next,path_dim,O_D_dim,d,Delta,Omega,t_a_0,n_a,cap,zeta[k,:])
            
            F_1_current = F_full_next[:path_dim]
            F_2_current = F_full_next[path_dim:]
            H_1_current = H_full_next[:path_dim]
            H_2_current = H_full_next[path_dim:]
            
            sum_tau += tau_glob
            sum_x_tau += tau_glob*x_next
            
            x_bar_prev = x_bar_curr
            x_bar_next = sum_x_tau/sum_tau
            x_bar_curr = x_bar_next
        return optim_gap, feasib_gap, iteration_time, x_bar_curr
        
        
        
            
            
        
        
    
    
    

