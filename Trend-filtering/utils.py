import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import cvxpy as cp
import mosek
from numpy import genfromtxt
import scipy.sparse as sparse
import random



def gen_data_leastsquares(N, n, r, pieces, sigma, myseed, const1=1, const2=0.1):
    
    np.random.seed(myseed)
    m = pieces
    A = np.random.randn(N,n)
    
    lengths = np.zeros(m)
    lengths[0:m-1] = round(n / m)
    lengths[m-1] = n - (m-1)*lengths[0]
    turns = np.cumsum(lengths)
    turns = turns.astype(int)
    lengths = lengths.astype(int)
    
    vals = 2*np.random.rand(m)-1
    vals = vals/np.linalg.norm(np.diff(vals),1)
    
    print("vals=", vals)
    
    x_true = np.zeros(n)
    
    if r == 1:
        x_true[0:turns[0]] = vals[0]
        for i in range(m-1):
            x_true[turns[i]:turns[i+1]] = vals[i+1]
        
        Q = np.zeros((n,1))
        Q[:,0] = 1/np.sqrt(n)
        x_true = x_true - Q@(Q.T@x_true) + const1
        
    
    if r == 2:
        x_true[0:turns[0]] = np.array([i * vals[0] for i in range(turns[0])])
        for i in range(m-1):
            x_true[turns[i]:turns[i+1]] = np.array([x_true[turns[i]-1] + vals[i+1]*(k+1) for k in range(lengths[i+1])])
        
        tmp = np.array([i for i in range(n)])
        tmp = tmp - np.mean(tmp)
        B = np.zeros((n,2))
        B[:,0] = 1
        B[:,1] = tmp
        Q = np.linalg.qr(B)[0]
        x_true = x_true - Q@(Q.T@x_true)
        x_true = x_true + const1 + const2*tmp
        
    
    b0 = A @ x_true
    eps = np.random.randn(N)*(sigma*np.linalg.norm(b0)/np.sqrt(N))
    b = b0 + eps
    b = np.reshape(b, (N,1))
    
    return A, b, x_true
    
    










