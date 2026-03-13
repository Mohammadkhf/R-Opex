# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 23:35:30 2025

@author: Mohamad
"""

import numpy as np
import pandas as pd

def mutual_information(x, y, base=2):
    """
    Compute mutual information I(X; Y) between two discrete variables x and y.
    
    Parameters
    ----------
    x : array-like (list, np.array, or pd.Series)
        First discrete variable.
    y : array-like
        Second discrete variable (same length as x).
    base : float
        Logarithm base (2 for bits, np.e for nats).
        
    Returns
    -------
    mi : float
        Mutual information I(X; Y).
    """
    x = pd.Series(x)
    y = pd.Series(y)
    
    # Joint frequency table
    joint_counts = pd.crosstab(x, y)
    
    # Convert to probabilities
    joint_prob = joint_counts / joint_counts.to_numpy().sum()
    
    # Marginal probabilities
    px = joint_prob.sum(axis=1)      # p(x)
    py = joint_prob.sum(axis=0)      # p(y)
    
    # Convert to numpy arrays for broadcasting
    pxy = joint_prob.to_numpy()
    pxv = px.to_numpy()[:, None]     # column vector
    pyv = py.to_numpy()[None, :]     # row vector
    
    # Avoid log(0): only keep entries where p(x,y) > 0
    mask = pxy > 0
    
    # I(X;Y) = sum p(x,y) * log( p(x,y) / (p(x)p(y)) )
    mi = (pxy[mask] * np.log(pxy[mask] / (pxv * pyv)[mask])).sum()
    
    # Change log base if needed
    if base != np.e:
        mi /= np.log(base)
    
    return mi
