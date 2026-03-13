# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 00:12:22 2025

@author: Mohamad
"""

# -*- coding: utf-8 -*-
"""
Main file of second setting of minimization problem with traffic equilibrium problem.
Runs 5 reps with n_a in {1.0, 1.5, 2.0, 2.5, 3.0} and plots requested slices.
"""

# %% imports
import numpy as np
import time
import matplotlib.pyplot as plt
from bilevel import part2
%matplotlib qt 

# %% config
K = 500_000                # total iterations
R = 5                      # number of replications (for the 5 n_a values)
n = 8
path_dim = 6
O_D_dim = 2
link_dim = 7

# check-pointing (every K/100 iters)
tresh_eval = max(1, K // 100)
num_pts = K // tresh_eval                      # number of checkpoints
idxs_IREG = np.arange(1, min(50, num_pts) + 1)  # 1..50
idxs_IRX  = np.arange(2, min(100, num_pts) + 1, 2)  # 2,4,..,100

# the 5 exponents for n_a (as requested 1,1.5,2,2.5 + added 3.0 to make 5 curves)
n_a_values = [1.0, 1.5, 2.0, 2.5, 3.0]

# storage per replication (no averaging — each rep is a distinct n_a)
optim_gap_IRopex  = np.zeros((R, num_pts + 1))
feasib_gap_IRopex = np.zeros((R, num_pts + 1))
optim_gap_IREG    = np.zeros((R, num_pts + 1))
feasib_gap_IREG   = np.zeros((R, num_pts + 1))

# %% constants that don't change across reps
xi        = np.zeros((2 * K, O_D_dim))
xi_test   = np.zeros((min(5_000_000, 10 * K), O_D_dim))
xi_test_avg = np.mean(xi_test, axis=0)
xi_test_square = np.mean(xi_test**2, axis=0)

# using ones (your uniform(1,1) was equivalent)
zeta         = np.ones((2 * K, path_dim))
zeta_test    = np.ones((min(5_000_000, 10 * K), path_dim))
zeta_test_avg = np.mean(zeta_test, axis=0)

t_a_0 = 1.0
cap = 40 * np.array([10, 10, 10, 10, 10, 10, 10], dtype=np.float64)
d = np.array([200, 220], dtype=np.float64)

Delta = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0]
], dtype=np.float64)

Omega = np.array([[1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1]], dtype=np.float64)

upper_path = np.array([200, 250, 350, 400, 500, 700], dtype=np.float64)
upper_O_D = Omega @ upper_path

var_obj = 0.0
var_inner = 0.0
L_H = 0.0
mu = 0  # monotone case

# L_F depends on n_a through c_flow_deriv; we’ll recompute inside the loop

# %% run reps for each n_a
for r, na_scalar in enumerate(n_a_values):
    # set exponent vector for this replication
    n_a = na_scalar * np.ones(link_dim)

    # Lipschitz constants (recompute due to n_a)
    c_flow_deriv = t_a_0 * 0.15 * n_a / cap * ((Delta @ upper_path) / cap) ** (n_a - 1)
    L_c = np.max(c_flow_deriv)
    L_F = np.sqrt(
        2 * (np.linalg.norm(Delta.T) * L_c * np.linalg.norm(Delta)) ** 2
        + np.linalg.norm(Omega) ** 2
        + np.linalg.norm(Omega.T) ** 2
    )

    # randomized init within bounds
    x_init = np.zeros(n)
    x_init[:path_dim] = np.random.uniform(0, upper_path)
    x_init[path_dim:] = np.random.uniform(0, upper_O_D)

    # --- IRopex
    (optim_gap_IRopex[r, :],
     feasib_gap_IRopex[r, :],
     _iter_time_irx,
     _xbar) = part2.IRopex(
        n, path_dim, O_D_dim, link_dim,
        tresh_eval,
        xi, xi_test, xi_test_avg,
        zeta, zeta_test, zeta_test_avg,
        n_a, t_a_0,
        cap, d, Delta, Omega, upper_path, var_obj, var_inner,
        L_F, L_H,
        mu, K, x_init, upper_O_D
    )

    # --- IREG
    (optim_gap_IREG[r, :],
     feasib_gap_IREG[r, :],
     _iter_time_ireg,
     _ybar) = part2.IREG(
        n, path_dim, O_D_dim, link_dim,
        tresh_eval,
        xi, xi_test, xi_test_avg,
        zeta, zeta_test, zeta_test_avg,
        n_a, t_a_0,
        cap, d, Delta, Omega, upper_path, var_obj, var_inner,
        L_F, L_H,
        mu, K, x_init, upper_O_D
    )

# %% plots (x-axis scaled with ×10^5 at the end, log y-scale)
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.legend_handler import HandlerLine2D

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
title_font = {'fontsize': 14, 'fontweight': 'bold'}
label_font = {'fontsize': 12}

# x-axis mapping to gradient evaluations
x_irx  = tresh_eval * idxs_IRX
x_ireg = 2 * tresh_eval * idxs_IREG

cmap = plt.cm.get_cmap("tab10", len(n_a_values))
colors = [cmap(i) for i in range(len(n_a_values))]

# ---------- OPTIMALITY ----------
ax[0].set_title("Optimality vs Gradient Evaluations", fontdict=title_font)
for r, na_scalar in enumerate(n_a_values):
    ax[0].plot(x_irx, optim_gap_IRopex[r, idxs_IRX],
               linestyle='-', marker='o', markersize=4, markevery=5,
               linewidth=2.2, color=colors[r],
               label=f"Ropex, n_a={na_scalar:g}")
    ax[0].plot(x_ireg, optim_gap_IREG[r, idxs_IREG],
               linestyle='--', marker='^', markersize=4, markevery=5,
               linewidth=1.8, color=colors[r],
               label=f"IREG, n_a={na_scalar:g}")

ax[0].set_xlabel("Gradient evaluations", fontdict=label_font)
ax[0].set_ylabel(r"$\|\bar{x}_{k+1} - \bar{x}_{k}\|$")
ax[0].set_yscale("log")   # <<< log scale

# ---------- FEASIBILITY ----------
ax[1].set_title("Feasibility vs Gradient Evaluations", fontdict=title_font)
for r, na_scalar in enumerate(n_a_values):
    ax[1].plot(x_irx, feasib_gap_IRopex[r, idxs_IRX],
               linestyle='-', marker='o', markersize=4, markevery=5,
               linewidth=2.2, color=colors[r],
               label=f"Ropex, n_a={na_scalar:g}")
    ax[1].plot(x_ireg, feasib_gap_IREG[r, idxs_IREG],
               linestyle='--', marker='^', markersize=4, markevery=5,
               linewidth=1.8, color=colors[r],
               label=f"IREG, n_a={na_scalar:g}")

ax[1].set_xlabel("Gradient evaluations", fontdict=label_font)
ax[1].set_ylabel(r"$\phi(\bar{x}_k)$")
ax[1].set_yscale("log")   # <<< log scale

# ---------- Single ×10^5 at the end ----------
for a in ax:
    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((0, 0))
    a.xaxis.set_major_formatter(sf)

# legends (longer handles so dashed shows)
ax[0].legend(ncol=2, fontsize=9, handlelength=3, numpoints=1,
             handler_map={plt.Line2D: HandlerLine2D(numpoints=2)})
ax[1].legend(ncol=2, fontsize=9, handlelength=3, numpoints=1,
             handler_map={plt.Line2D: HandlerLine2D(numpoints=2)})

plt.tight_layout()
plt.show()



