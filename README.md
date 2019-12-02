 Frank-Wolfe-with-unbounded-constraints
   The project consists of experiments of CGG on two problems: Trend filtering and matrix completion with side information.

   The directory 'Trend_filtering' contains four experiments and all other functions used therein.

  The first and second experiments are 'Exp1_Comparison_Plotting_LeastSquare' and 'Exp2_Comparison_Plotting_Logistic', which 
are comparisons of the original algorithms and transformed algorithms, on least square and logistic objective functions respectively. Two figures which show the performance of different algorithms will be outputted and saved. Latex codes to include these figures will be automatically generated and saved in documents in the repisitory 'TexFile'.

  The third experiment is 'Exp3_FixedTolLeastSquare', which reports the time for CGG, Awaystep-CGG and Mosek on different sizes and different choices of parameters for least square objtective functions. It will output a table containing the results, whose Latex code can be found in documents in the repisitory 'TexFile'.

  The fourth experiment is 'Exp4_LogisticPath', which solve the trend filtering for logistic loss functions along a path of different values of delta. 
  
  The directory 'Matrix_completion_with_side_informationg' contains one experiment 'Exp1' and all other functions used therein. It compares the speed of CGG and SCS for different choices of parameters. 'MCside_func' is the main function used by 'Exp1'. 'SCS_SIMC' and 'CGG_SIMC' are functions to use the SCS and CGG to compute the solution. They are used in 'MCside_func'. Functions 'Afun' and 'Afunc' are used to defining the function handle to use the svds. 
