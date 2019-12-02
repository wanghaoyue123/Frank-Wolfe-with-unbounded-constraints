import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import cvxpy as cp
import mosek
from numpy import genfromtxt
import scipy.sparse as sparse

def generate_data_leastsquare(A, r, N, n, sigma):
    ############################################################
    ## Generate piecewise constant (r=1) or piecewise linear (r=2) data with 5 pieces before noising.
    ##
    ## Input:
    ## A: the desian matrix
    ## r: the order of difference.
    ## N: the number of samples
    ## n: the number of features
    ## sigma: the noise level
    ##
    ## Author: Haoyue Wang
    ## Email: haoyuew@mit.edu
    ## Date: 2019. 11. 24
    ## Reference:
    ############################################################


    bar_x = np.zeros((n,))
    len1 = round(n / 6)
    len2 = round(n / 5)
    len3 = round(n / 4)
    len4 = round(n / 4)
    len5 = n - len1 - len2 - len3 - len4

    turn1 = len1
    turn2 = turn1 + len2
    turn3 = turn2 + len3
    turn4 = turn3 + len4
    turn5 = turn4 + len5

    slope1 = 0.4
    slope2 = 0
    slope3 = -0.24
    slope4 = 0
    slope5 = -0.12

    # A = np.random.randn(N, n)

    if r == 1:
        ## r = 1 data
        ## delta = 1
        bar_x[0:turn1] = np.array([2 / 5 for i in range(turn1)])
        bar_x[turn1:turn2] = np.array([1 / 5 for i in range(len2)])
        bar_x[turn2:turn3] = np.array([2 / 5 for i in range(len3)])
        bar_x[turn3:turn4] = np.array([3 / 5 for i in range(len4)])
        bar_x[turn4:turn5] = np.array([1 / 5 for i in range(len5)])
        bar_x = np.reshape(bar_x, (n, 1))
    else:
        if r == 2:
            ## r = 2 data
            ## delta = 1
            bar_x[0:turn1] = np.array([i * slope1 for i in range(turn1)])
            bar_x[turn1:turn2] = np.array([bar_x[turn1 - 1] + slope2 * i for i in range(len2)])
            bar_x[turn2:turn3] = np.array([bar_x[turn2 - 1] + slope3 * i for i in range(len3)])
            bar_x[turn3:turn4] = np.array([bar_x[turn3 - 1] + slope4 * i for i in range(len4)])
            bar_x[turn4:turn5] = np.array([bar_x[turn4 - 1] + slope5 * i for i in range(len5)])
            bar_x = np.reshape(bar_x, (n, 1))

    b = A @ bar_x
    eps = np.random.randn(N,1)
    b_blured = b + sigma * eps * np.linalg.norm(b) / np.sqrt(N)
    b = b_blured
    return b


def generate_data_logistic(X, r, N, n, sigma):
    beta = np.zeros((n,))
    len1 = round(n / 6)
    len2 = round(n / 5)
    len3 = round(n / 4)
    len4 = round(n / 4)
    len5 = n - len1 - len2 - len3 - len4

    turn1 = len1
    turn2 = turn1 + len2
    turn3 = turn2 + len3
    turn4 = turn3 + len4
    turn5 = turn4 + len5

    slope1 = 0.4
    slope2 = 0
    slope3 = -0.24
    slope4 = 0
    slope5 = -0.12

    if r == 1:
        ## r = 1 data
        ## delta = 1
        beta[0:turn1] = np.array([2 / 5 for i in range(turn1)])
        beta[turn1:turn2] = np.array([1 / 5 for i in range(len2)])
        beta[turn2:turn3] = np.array([2 / 5 for i in range(len3)])
        beta[turn3:turn4] = np.array([3 / 5 for i in range(len4)])
        beta[turn4:turn5] = np.array([1 / 5 for i in range(len5)])
        beta = np.reshape(beta, (n, 1))
    else:
        if r == 2:
            ## r = 2 data
            ## delta = 1
            beta[0:turn1] = np.array([i * slope1 for i in range(turn1)])
            beta[turn1:turn2] = np.array([beta[turn1 - 1] + slope2 * i for i in range(len2)])
            beta[turn2:turn3] = np.array([beta[turn2 - 1] + slope3 * i for i in range(len3)])
            beta[turn3:turn4] = np.array([beta[turn3 - 1] + slope4 * i for i in range(len4)])
            beta[turn4:turn5] = np.array([beta[turn4 - 1] + slope5 * i for i in range(len5)])
            beta = np.reshape(beta, (n, 1))

    # X = np.random.randn(N, n)
    prob = 1 / (1 + np.exp(-sigma * X @ beta))
    ran = np.random.random((N, 1))
    y = (ran - prob) > 0
    y = y * 2 - 1
    A = -sigma * X * y
    return A
