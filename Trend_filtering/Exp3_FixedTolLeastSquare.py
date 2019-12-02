import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from Data_generation import generate_data_leastsquare
from CVXPY_Leastsquare import CVXPY_Leastsquare
from Original_CGG_Leastsquare import Original_CGG_Leastsquare
from Original_AwaystepCGG_Leastsquare import Original_AwaystepCGG_Leastsquare
from Transformed_CGG_Leastsquare import Transformed_CGG_Leastsquare
from Transformed_AwaystepCGG_Leastsquare import Transformed_AwaystepCGG_Leastsquare
from CGF_Leastsquare import CGF_Leastsquare

from CVXPY_Logistic import  CVXPY_Logistic
from Original_CGG_Logistic import  Original_CGG_Logistic
from Original_AwaystepCGG_Logistic import Original_AwaystepCGG_Logistic
from Transformed_CGG_Logistic import Transformed_CGG_Logistic
from Transformed_AwaystepCGG_Logistic import  Transformed_AwaystepCGG_Logistic

def LeastSquare_fixed_tol(r, N, n, delta, sigma, itermax, tolerance, repeat):
    #####################################################
    # Report the time for CGG-simple, CGG-awaystep to reach a fixed tolerance
    #####################################################
    ave_time_cvxpy = 0
    ave_time1 = 0
    ave_time2 = 0
    fail1 = 0
    fail2 = 0
    A = np.random.randn(N,n)
    for k in range(repeat):
        b = generate_data_leastsquare(A, r=r, N=N, n=n, sigma=sigma)
        time_cvxpy, obj_cvxpy = CVXPY_Leastsquare(r, A, b, delta)
        time1, obj1 = Original_CGG_Leastsquare(r=r, A=A, b=b, delta=delta, step_size='simple', itermax=itermax)
        time2, obj2 = Original_AwaystepCGG_Leastsquare(r=r, A=A, b=b, delta=delta, itermax=itermax)

        gap1 = np.reshape((obj1 - obj_cvxpy) / (1 + obj_cvxpy), (itermax,))
        gap2 = np.reshape((obj2 - obj_cvxpy) / (1 + obj_cvxpy), (itermax,))

        if np.min(gap1) < tolerance:
            a1 = np.nonzero(gap1 < tolerance)
            first_index1 = a1[0][0]
            first_time1 = time1[first_index1, 0]
            ave_time1 = ave_time1 + first_time1
        else:
            fail1 = 1

        if np.min(gap2) < tolerance:
            a2 = np.nonzero(gap2 < tolerance)
            first_index2 = a2[0][0]
            first_time2 = time2[first_index2, 0]
            ave_time2 = ave_time2 + first_time2
        else:
            fail2 = 1

        ave_time_cvxpy = ave_time_cvxpy + time_cvxpy

    if fail1 == 1:
        ave_time1 = -1
    else:
        ave_time1 = ave_time1 / repeat

    if fail2 == 1:
        ave_time2 = -1
    else:
        ave_time2 = ave_time2 / repeat

    ave_time_cvxpy = ave_time_cvxpy / repeat
    
    return [ave_time1, ave_time2, ave_time_cvxpy]





## Experiment 1: report the time for CGG and Away-step CGG to reach a certain tolerance
tol = 5e-4
Data_table = np.zeros((12, 3))
repeat = 3

r1 = 1
r2 = 2
N1 = 500
n1 = 20000
N2 = 5000
n2 = 5000
N3 = 10000
n3 = 1000
delta1 = 0.8
delta2 = 1.2
sigma1 = 0.5
sigma2 = 1
itermax = 10000

# Data_table[0, :] = LeastSquare_fixed_tol(r=r1, N=N1, n=n1, delta=delta1, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
# Data_table[1, :] = LeastSquare_fixed_tol(r=r1, N=N1, n=n1, delta=delta1, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)
# Data_table[2, :] = LeastSquare_fixed_tol(r=r1, N=N1, n=n1, delta=delta2, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
# Data_table[3, :] = LeastSquare_fixed_tol(r=r1, N=N1, n=n1, delta=delta2, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)
#
# Data_table[4, :] = LeastSquare_fixed_tol(r=r1, N=N2, n=n2, delta=delta1, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
# Data_table[5, :] = LeastSquare_fixed_tol(r=r1, N=N2, n=n2, delta=delta1, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)
# Data_table[6, :] = LeastSquare_fixed_tol(r=r1, N=N2, n=n2, delta=delta2, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
# Data_table[7, :] = LeastSquare_fixed_tol(r=r1, N=N2, n=n2, delta=delta2, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)
#
# Data_table[8, :] = LeastSquare_fixed_tol(r=r1, N=N3, n=n3, delta=delta1, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
# Data_table[9, :] = LeastSquare_fixed_tol(r=r1, N=N3, n=n3, delta=delta1, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)
# Data_table[10, :] = LeastSquare_fixed_tol(r=r1, N=N3, n=n3, delta=delta2, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
# Data_table[11, :] = LeastSquare_fixed_tol(r=r1, N=N3, n=n3, delta=delta2, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)

Data_table[0, :] = LeastSquare_fixed_tol(r=r2, N=N1, n=n1, delta=delta1, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
Data_table[1, :] = LeastSquare_fixed_tol(r=r2, N=N1, n=n1, delta=delta1, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)
Data_table[2, :] = LeastSquare_fixed_tol(r=r2, N=N1, n=n1, delta=delta2, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
Data_table[3, :] = LeastSquare_fixed_tol(r=r2, N=N1, n=n1, delta=delta2, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)

Data_table[4, :] = LeastSquare_fixed_tol(r=r2, N=N2, n=n2, delta=delta1, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
Data_table[5, :] = LeastSquare_fixed_tol(r=r2, N=N2, n=n2, delta=delta1, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)
Data_table[6, :] = LeastSquare_fixed_tol(r=r2, N=N2, n=n2, delta=delta2, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
Data_table[7, :] = LeastSquare_fixed_tol(r=r2, N=N2, n=n2, delta=delta2, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)

Data_table[8, :] = LeastSquare_fixed_tol(r=r2, N=N3, n=n3, delta=delta1, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
Data_table[9, :] = LeastSquare_fixed_tol(r=r2, N=N3, n=n3, delta=delta1, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)
Data_table[10, :] = LeastSquare_fixed_tol(r=r2, N=N3, n=n3, delta=delta2, sigma=sigma1, itermax=itermax, tolerance=tol, repeat=repeat)
Data_table[11, :] = LeastSquare_fixed_tol(r=r2, N=N3, n=n3, delta=delta2, sigma=sigma2, itermax=itermax, tolerance=tol, repeat=repeat)


Data_frame = pd.DataFrame(Data_table )
Data_frame.columns = ('CGG_simple', 'Awaystep_CGG',
                        'Mosek')
Data_frame.index = (
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' %(r1,N1,n1,delta1,sigma1),
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' %(r1,N1,n1,delta1,sigma2),
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' %(r1,N1,n1,delta2,sigma1),
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' %(r1,N1,n1,delta2,sigma2),
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' %(r1,N2,n2,delta1,sigma1),
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' %(r1,N2,n2,delta1,sigma2),
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' %(r1,N2,n2,delta2,sigma1),
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' %(r1,N2,n2,delta2,sigma2),
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r1, N3, n3, delta1, sigma1),
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r1, N3, n3, delta1, sigma2),
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r1, N3, n3, delta2, sigma1),
#                     'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r1, N3, n3, delta2, sigma2),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N1, n1, delta1, sigma1),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N1, n1, delta1, sigma2),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N1, n1, delta2, sigma1),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N1, n1, delta2, sigma2),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N2, n2, delta1, sigma1),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N2, n2, delta1, sigma2),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N2, n2, delta2, sigma1),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N2, n2, delta2, sigma2),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N3, n3, delta1, sigma1),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N3, n3, delta1, sigma2),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N3, n3, delta2, sigma1),
                    'r=%d, (N,n)=(%d,%d), delta=%.1f, sigma=%.1f' % (r2, N3, n3, delta2, sigma2),
                    )
Table_latex = Data_frame.to_latex(float_format='%.3f')
# print(Table_latex)
fo = open('TexFile/Exp3_Fixed_tol_LeastSquare.tex', 'a')
fo.write('\n')
fo.write('\\begin{table}[H]\n')
fo.write('\\centering\n')
fo.write('\\label{Table_exp1}\n')
fo.write(Table_latex)
fo.write('\\caption{Experiment1 on trend filtering with quadratic loss: The CPU time of CGG with simple step size and Awaystep-CGG'
         'to reach a fixed tolerance %.1e of the relative gap. The fourth column reports the '
         'time required by Mosek. All the reported results are the average of %d independent experiments'
         'The value -1 means that in at least one of the %d experiments, the algorithm doesnot reach the tolerance %.1e within %d iterations'
         '}\n' %(tol,repeat,repeat, tol, itermax))
fo.write('\\end{table}\n')
fo.write('\n')
fo.write('\n')