import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data_generation import generate_data_logistic
from CVXPY_Logistic import  CVXPY_Logistic
from Original_CGG_Logistic import Original_CGG_Logistic
from Original_AwaystepCGG_Logistic import Original_AwaystepCGG_Logistic
from Transformed_CGG_Logistic import Transformed_CGG_Logistic
from Transformed_AwaystepCGG_Logistic import Transformed_AwaystepCGG_Logistic





def my_plotting_Logistic(Img_ID, r, N, n, relative_delta, sigma, repeat, itermax = 2000):
    #####################################################################
    ## Plot the experiments results on trend filtering with logistic loss, and generate the latex code for the figure.
    ## All the images are saved in a sub-directory named 'Images'.
    ## The name of the figures are 'Logistic_Iterations_vs_RelativeGap_Img_ID.eps' and 'Logistic_Time_vs_RelativeGap_Img_ID.eps'
    ## All the 'includegraphics' code for latex and discriptions of the setting are saved in 'TexFile/ExpFigures_Logistic.tex'
    ##
    ## Parameters:
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
    ## sigma: Noise level (See captions of plotted figures in the latex code).
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
        piece_type = 'constant'
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

    relative_gap11_ave = 0
    relative_gap12_ave = 0
    relative_gap2_ave = 0
    relative_gap31_ave = 0
    relative_gap32_ave = 0
    relative_gap4_ave = 0


    for j in range(repeat):
        X = np.random.randn(N, n)
        A = generate_data_logistic(X, r, N, n, sigma)

        ## Output the cpu time and objective values in each iteration for different methods
        time_cvxpy, cvxpy_opt = CVXPY_Logistic(r, A, delta)

        # original_CGG(r, A, b, delta, step_size, *itermax, *cache_length)
        time_vec11, obj_val11 = Original_CGG_Logistic(r, A, delta, 'simple', itermax = itermax)
        time_vec12, obj_val12 = Original_CGG_Logistic(r, A, delta, 'linesearch', itermax = itermax)
        time_vec2, obj_val2 = Original_AwaystepCGG_Logistic(r, A, delta, itermax = itermax)
        time_vec31, obj_val31 = Transformed_CGG_Logistic(r, A, delta, 'simple', itermax = itermax)
        time_vec32, obj_val32 = Transformed_CGG_Logistic(r, A, delta, 'linesearch', itermax = itermax)
        time_vec4, obj_val4 = Transformed_AwaystepCGG_Logistic(r, A, delta, itermax = itermax)

        # Relative gap
        relative_gap11 = (obj_val11 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap12 = (obj_val12 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap2 = (obj_val2 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap31 = (obj_val31 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap32 = (obj_val32 - cvxpy_opt)/(1 + cvxpy_opt)
        relative_gap4 = (obj_val4 - cvxpy_opt)/(1 + cvxpy_opt)

        relative_gap11_ave = relative_gap11_ave + relative_gap11
        relative_gap12_ave = relative_gap12_ave + relative_gap12
        relative_gap2_ave = relative_gap2_ave + relative_gap2
        relative_gap31_ave = relative_gap31_ave + relative_gap31
        relative_gap32_ave = relative_gap32_ave + relative_gap32
        relative_gap4_ave = relative_gap4_ave + relative_gap4

        time_vec11_ave = time_vec11_ave + time_vec11
        time_vec12_ave = time_vec12_ave + time_vec12
        time_vec2_ave = time_vec2_ave + time_vec2
        time_vec31_ave = time_vec31_ave + time_vec31
        time_vec32_ave = time_vec32_ave + time_vec32
        time_vec4_ave = time_vec4_ave + time_vec4

    time_vec11_ave = time_vec11_ave / repeat
    time_vec12_ave = time_vec12_ave / repeat
    time_vec2_ave = time_vec2_ave / repeat
    time_vec31_ave = time_vec31_ave / repeat
    time_vec32_ave = time_vec32_ave / repeat
    time_vec4_ave = time_vec4_ave / repeat

    relative_gap11_ave = relative_gap11_ave / repeat
    relative_gap12_ave = relative_gap12_ave / repeat
    relative_gap2_ave = relative_gap2_ave / repeat
    relative_gap31_ave = relative_gap31_ave / repeat
    relative_gap32_ave = relative_gap32_ave / repeat
    relative_gap4_ave = relative_gap4_ave / repeat

    ## Plot a point every (itermax / 20) iterations
    plot_gap = int(itermax / 20)
    plot_index = [plot_gap * i for i in range(20)]
    relative_gap11_plot = relative_gap11_ave[plot_index]
    relative_gap12_plot = relative_gap12_ave[plot_index]
    relative_gap2_plot = relative_gap2_ave[plot_index]
    relative_gap31_plot = relative_gap31_ave[plot_index]
    relative_gap32_plot = relative_gap32_ave[plot_index]
    relative_gap4_plot = relative_gap4_ave[plot_index]

    time_vec11_plot = time_vec11_ave[plot_index]
    time_vec12_plot = time_vec12_ave[plot_index]
    time_vec2_plot = time_vec2_ave[plot_index]
    time_vec31_plot = time_vec31_ave[plot_index]
    time_vec32_plot = time_vec32_ave[plot_index]
    time_vec4_plot = time_vec4_ave[plot_index]


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
    plt.legend()

    plt.gca().set_yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Relative gap')
    plt.savefig('Images/Logistic_Iterations_vs_RelativeGap_%d.eps' %(Img_ID))
    plt.show()


    ## Plot 2: CPU time vs Relative gap

    plt.figure()
    plt.plot(time_vec11_plot, relative_gap11_plot,'>-', label='orCGG_simple')
    plt.plot(time_vec12_plot, relative_gap12_plot, 'x-', label='orCGG_linesearch')
    plt.plot(time_vec2_plot,relative_gap2_plot, 'o-', label='orAwayCGG')
    plt.plot(time_vec31_plot,relative_gap31_plot, 's-', label='trCGG_simple')
    plt.plot(time_vec32_plot,relative_gap32_plot, '*-', label='trCGG_linesearch')
    plt.plot(time_vec4_plot,relative_gap4_plot, 'v-', label='trAwayCGG')
    plt.legend()

    plt.gca().set_yscale('log')
    plt.xlabel('cpu time (seconds)')
    plt.ylabel('Relative gap')
    plt.savefig('Images/Logistic_Time_vs_RelativeGap_%d.eps' %(Img_ID))
    plt.show()



    ## Include the figure into latex code
    fo = open('TexFile/Exp2_Figures_Logistic_%d.tex' %(Img_ID), 'a')
    fo.write('\n')
    fo.write('\\begin{figure}[htbp]\n')
    fo.write('\\centering\n')
    fo.write('\\includegraphics[height=7.2cm,width=9cm]{Images/Logistic_Iterations_vs_RelativeGap_%d.eps}\n' %(Img_ID))
    fo.write('\\caption{Iteration-Gap figure for trend filtering with logistic loss '
             '$\\sum_{i=1}^N \\log \\left( 1+\\exp(-y_i\\sigma x_i^T \\beta) \\right)$ '
             'and constraint $\\|D^{(r)} \\beta\\|_1 \\le \\delta$, '
             'where $\\beta\\in \\R^n$ with $N= %d$, $n = %d$, $r = %d$, $\\delta= %.1f$ and $\\sigma= %.1e$. '
             '$y_1,...,y_N$ are i.i.d. samples generated from the distribution '
             '$\\mathbb{P}(y_i = 1) = \\frac{1}{1+\\exp(-\\sigma x_i^T \\bar \\beta)}, $ '
             '$\\mathbb{P}(y_i = -1) = 1- \\mathbb{P}(y_i = 1)$, where $\\bar \\beta$ is piecewise %s with 5 pieces and $\\|D^{(r)} \\bar \\beta\\|_1 = 1$.'
             '}\n'
             %(N, n, r, delta, sigma, piece_type))
    fo.write('\\label{Iteration-Gap%d}\n' %(Img_ID))
    fo.write('\\end{figure}')
    fo.write('\n')
    fo.write('\n')


    fo.write('\n')
    fo.write('\\begin{figure}[htbp]\n')
    fo.write('\\centering\n')
    fo.write('\\includegraphics[height=7.2cm,width=9cm]{Images/Logistic_Time_vs_RelativeGap_%d.eps}\n' %(Img_ID))
    fo.write('\\caption{Time-Gap figure for trend filtering with logistic loss '
             '$\\sum_{i=1}^N \\log \\left( 1+\\exp(-y_i\\sigma x_i^T \\beta) \\right)$ '
             'and constraint $\\|D^{(r)} \\beta\\|_1 \\le \\delta$, '
             'where $\\beta\\in \\R^n$ with $N= %d$, $n = %d$, $r = %d$, $\\delta= %.1f$ and $\\sigma= %.1e$. '
             '$y_1,...,y_N$ are i.i.d. samples generated from the distribution '
             '$\\mathbb{P}(y_i = 1) = \\frac{1}{1+\\exp(-\\sigma x_i^T \\bar \\beta)}, $ '
             '$\\mathbb{P}(y_i = -1) = 1- \\mathbb{P}(y_i = 1)$, where $\\bar \\beta$ is piecewise %s with 5 pieces and $\\|D^{(r)} \\bar \\beta\\|_1 = 1$. '
             'The time taken by CVXPY is %f seconds.'
             '}\n'
             %(N, n, r, delta, sigma, piece_type, time_cvxpy))
    fo.write('\\label{Time-Gap%d}\n' %(Img_ID))
    fo.write('\\end{figure}')
    fo.write('\n')
    fo.write('\n')

    fo.close()



my_plotting_Logistic(Img_ID= 7, r= 2, N=200, n=400, relative_delta= 0.8, sigma= 0.02, repeat=3, itermax = 3000)