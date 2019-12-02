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


def my_plotting_LeastSquare(Img_ID, r, N, n, relative_delta, relative_sigma, repeat, itermax = 2000):
    #####################################################################
    ## Plot the experiments results on trend filtering with quadratic loss, and generate the latex code for the figure.
    ## All the images are saved in a sub-directory named 'Images'.
    ## All the 'includegraphics' code for latex and discriptions of the setting are saved in 'TexFile/ExpFigures_LeastSquare.tex'
    ##
    ## Input:
    ##
    ## Img_ID: Int type.
    ## This parameter is used to give different names when saving the images.
    ##
    ## r: The order of difference for D^(r).
    ##
    ## N: The number of samples.
    ##
    ## n: The number of features.
    ##
    ## relative_delta: Relative value w.r.t. the genrated data beforing noising. (See captions of plotted figures in the latex code).
    ##
    ## relative_sigma: Relative noise level (See captions of plotted figures in the latex code).
    ##
    ## repeat: The number of independent experiments.
    ##
    ## itermax: The total number of iterations for each method.
    ##
    ##
    ## Author: Haoyue Wang
    ## Email: haoyuew@mit.edu
    ## Date: 2019. 11. 24
    ## Reference:
    ####################################################################


    if r == 1:
        delta = relative_delta * 1
        piece_type = 'contant'
    else:
        if r == 2:
            delta = relative_delta * 1
            piece_type = 'linear'

    time_vec11_ave = 0
    time_vec12_ave = 0
    time_vec2_ave = 0
    time_vec31_ave = 0
    time_vec32_ave = 0
    time_vec4_ave = 0
    time_vec51_ave = 0
    time_vec52_ave = 0

    relative_gap11_ave = 0
    relative_gap12_ave = 0
    relative_gap2_ave = 0
    relative_gap31_ave = 0
    relative_gap32_ave = 0
    relative_gap4_ave = 0
    relative_gap51_ave = 0
    relative_gap52_ave = 0



    for j in range(repeat):
        A = np.random.randn(N,n)
        b = generate_data_leastsquare(A, r, N, n, relative_sigma)

        ## Output the cpu time and objective values in each iteration for different methods
        time_cvxpy, cvxpy_opt = CVXPY_Leastsquare(r, A, b, delta)

        # original_CGG(r, A, b, delta, step_size, *itermax, *cache_length)
        time_vec11, obj_val11 = Original_CGG_Leastsquare(r, A, b, delta, 'simple', itermax=itermax)
        time_vec12, obj_val12 = Original_CGG_Leastsquare(r, A, b, delta, 'linesearch', itermax=itermax)
        time_vec2, obj_val2 = Original_AwaystepCGG_Leastsquare(r, A, b, delta, itermax=itermax)
        time_vec31, obj_val31 = Transformed_CGG_Leastsquare(r, A, b, delta, 'simple', itermax=itermax)
        time_vec32, obj_val32 = Transformed_CGG_Leastsquare(r, A, b, delta, 'linesearch', itermax=itermax)
        time_vec4, obj_val4 = Transformed_AwaystepCGG_Leastsquare(r, A, b, delta, itermax=itermax)
        time_vec51, obj_val51 = CGF_Leastsquare(r, A, b, delta, 'simple', itermax=itermax)
        time_vec52, obj_val52 = CGF_Leastsquare(r, A, b, delta, 'linesearch', itermax=itermax)

        relative_gap11 = (obj_val11 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap12 = (obj_val12 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap2 = (obj_val2 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap31 = (obj_val31 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap32 = (obj_val32 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap4 = (obj_val4 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap51 = (obj_val51 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap52 = (obj_val52 - cvxpy_opt)/(1 + cvxpy_opt)

        relative_gap11_ave = relative_gap11_ave + relative_gap11
        relative_gap12_ave = relative_gap12_ave + relative_gap12
        relative_gap2_ave = relative_gap2_ave + relative_gap2
        relative_gap31_ave = relative_gap31_ave + relative_gap31
        relative_gap32_ave = relative_gap32_ave + relative_gap32
        relative_gap4_ave = relative_gap4_ave + relative_gap4
        relative_gap51_ave = relative_gap51_ave + relative_gap51
        relative_gap52_ave = relative_gap52_ave + relative_gap52

        time_vec11_ave = time_vec11_ave + time_vec11
        time_vec12_ave = time_vec12_ave + time_vec12
        time_vec2_ave = time_vec2_ave + time_vec2
        time_vec31_ave = time_vec31_ave + time_vec31
        time_vec32_ave = time_vec32_ave + time_vec32
        time_vec4_ave = time_vec4_ave + time_vec4
        time_vec51_ave = time_vec51_ave + time_vec51
        time_vec52_ave = time_vec52_ave + time_vec52

    time_vec11_ave = time_vec11_ave / repeat
    time_vec12_ave = time_vec12_ave / repeat
    time_vec2_ave = time_vec2_ave / repeat
    time_vec31_ave = time_vec31_ave / repeat
    time_vec32_ave = time_vec32_ave / repeat
    time_vec4_ave = time_vec4_ave / repeat
    time_vec51_ave = time_vec51_ave / repeat
    time_vec52_ave = time_vec52_ave / repeat

    relative_gap11_ave = relative_gap11_ave / repeat
    relative_gap12_ave = relative_gap12_ave / repeat
    relative_gap2_ave = relative_gap2_ave / repeat
    relative_gap31_ave = relative_gap31_ave / repeat
    relative_gap32_ave = relative_gap32_ave / repeat
    relative_gap4_ave = relative_gap4_ave / repeat
    relative_gap51_ave = relative_gap51_ave / repeat
    relative_gap52_ave = relative_gap52_ave / repeat

    ## Plot a point every (itermax/20) iterations
    plot_gap = int(itermax/20)
    plot_index = [plot_gap * i for i in range(20)]
    relative_gap11_plot = relative_gap11_ave[plot_index]
    relative_gap12_plot = relative_gap12_ave[plot_index]
    relative_gap2_plot = relative_gap2_ave[plot_index]
    relative_gap31_plot = relative_gap31_ave[plot_index]
    relative_gap32_plot = relative_gap32_ave[plot_index]
    relative_gap4_plot = relative_gap4_ave[plot_index]
    relative_gap51_plot = relative_gap51_ave[plot_index]
    relative_gap52_plot = relative_gap52_ave[plot_index]

    time_vec11_plot = time_vec11_ave[plot_index]
    time_vec12_plot = time_vec12_ave[plot_index]
    time_vec2_plot = time_vec2_ave[plot_index]
    time_vec31_plot = time_vec31_ave[plot_index]
    time_vec32_plot = time_vec32_ave[plot_index]
    time_vec4_plot = time_vec4_ave[plot_index]
    time_vec51_plot = time_vec51_ave[plot_index]
    time_vec52_plot = time_vec52_ave[plot_index]




    ### Start plotting
    ## Plot 1: Iterations vs Relative gap
    x_axis = plot_index
    plt.figure()
    plt.plot(x_axis,relative_gap11_plot,'>-', label='orCGG_simple')
    plt.plot(x_axis,relative_gap12_plot,'x-', label='orCGG_linesearch')
    plt.plot(x_axis,relative_gap2_plot,'o-', label='orAwayCGG')
    plt.plot(x_axis,relative_gap31_plot,'s-', label='trCGG_simple')
    plt.plot(x_axis,relative_gap32_plot,'*-', label='trCGG_linesearch')
    plt.plot(x_axis,relative_gap4_plot,'v-', label='trAwayCGG')
    plt.plot(x_axis,relative_gap51_plot,'.-', label='CGF_simple')
    plt.plot(x_axis,relative_gap52_plot,'|-', label='CGF_linesearch')
    plt.legend()


    plt.gca().set_yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Relative gap')
    plt.savefig('Images/Leastsquare_Iterations_vs_RelativeGap_%d.eps' %(Img_ID))
    plt.show()

    ## Plot 2: CPU time vs Relative gap

    plt.figure()
    plt.plot(time_vec11_plot, relative_gap11_plot,'>-', label='orCGG_simple')
    plt.plot(time_vec12_plot, relative_gap12_plot, 'x-', label='orCGG_linesearch')
    plt.plot(time_vec2_plot,relative_gap2_plot, 'o-', label='orAwayCGG')
    plt.plot(time_vec31_plot,relative_gap31_plot, 's-', label='trCGG_simple')
    plt.plot(time_vec32_plot,relative_gap32_plot, '*-', label='trCGG_linesearch')
    plt.plot(time_vec4_plot,relative_gap4_plot, 'v-', label='trAwayCGG')
    plt.plot(time_vec51_plot,relative_gap51_plot, '.-', label='CGF_simple')
    plt.plot(time_vec52_plot,relative_gap52_plot, '|-', label='CGF_linesearch')
    plt.legend()

    plt.gca().set_yscale('log')
    plt.xlabel('cpu time (seconds)')
    plt.ylabel('Relative gap')
    plt.savefig('Images/Leastsquare_Time_vs_RelativeGap_%d.eps' %(Img_ID))
    plt.show()


    ## Output the Iteration-Gap figure:
    fo = open('TexFile/Exp1_Figures_LeastSquare_%d.tex' %(Img_ID), 'a')
    fo.write('\n')
    fo.write('\\begin{figure}[htbp]\n')
    fo.write('\\centering\n')
    fo.write('\\includegraphics[height=7.2cm,width=9cm]{Images/Leastsquare_Iterations_vs_RelativeGap_%d.eps}\n' %(Img_ID))
    fo.write('\\caption{Iteration-Gap figure for trend filtering with quadratic loss $\\| Ax-b\\|^2$ '
             'and constraint $\\|D^{(r)}\\| \\le \\delta$, '
             'where $A\\in \\R^{N\\times n}$, $r=%d$, $N=%d$, $n=%d$ and $\\delta= %.1f$. '
             'b is generated by $b = A\\bar x +\\epsilon$ with $\\bar x$ being '
             'piecewise %s with 5 pieces and $\\|D^{(%d)} \\bar x\\|_1 = 1$. '
             '$\\epsilon \\sim N(0, \\sigma^2)$ with noise $\\sigma= %.1f \\|A\\bar x\\|_2 / \\sqrt{N}$.   }\n'
             %(r, N, n, relative_delta, piece_type, r, relative_sigma))
    fo.write('\\label{Iteration-Gap%d}\n' %(Img_ID))
    fo.write('\\end{figure}')
    fo.write('\n')
    fo.write('\n')

    ## Output the Time-Gap figure:
    fo.write('\n')
    fo.write('\\begin{figure}[htbp]\n')
    fo.write('\\centering\n')
    fo.write('\\includegraphics[height=7.2cm,width=9cm]{Images/Leastsquare_Time_vs_RelativeGap_%d.eps}\n' %(Img_ID))
    fo.write('\\caption{Time-Gap figure for trend filtering with quadratic loss $\\| Ax-b\\|^2$ '
             'and constraint $\\|D^{(r)}\\| \\le \\delta$, '
             'where $A\\in \\R^{N\\times n}$, $r=%d$, $N=%d$, $n=%d$ and $\\delta= %.1f$. '
             'b is generated by $b = A\\bar x +\\epsilon$ with $\\bar x$ being '
             'piecewise %s with 5 pieces and $\\|D^{(%d)} \\bar x\\|_1 = 1$. '
             '$\\epsilon \\sim N(0, \\sigma^2)$ with noise $\\sigma= %.1f \\|A\\bar x\\|_2 / \\sqrt{N}$. '
             'The time taken by CVXPY is %f seconds. }\n'
             %(r, N, n, relative_delta, piece_type, r, relative_sigma, time_cvxpy))
    fo.write('\\label{Time-Gap%d}\n' %(Img_ID))
    fo.write('\\end{figure}')
    fo.write('\n')
    fo.write('\n')

    fo.close()


my_plotting_LeastSquare(Img_ID= 6, r= 1, N=400, n=400, relative_delta= 0.8, relative_sigma= 0.2, repeat = 3, itermax = 3000)
#my_plotting_LeastSquare(Img_ID= 5, r= 1, N=200, n=400, relative_delta= 0.8, relative_sigma= 0.2, itermax = 3000)

