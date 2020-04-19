import numpy as np
import scipy.optimize as sopt

#from util import *

#Lagrange Dual Function to be maximized w/r/t lambda_vars
#Returns negative value because we want to maximize it using a minimization function

#WARNING: if solvr_obj = None, then X,S,c_const != None
def lagrange_dual_func(lambda_vars, solvr_obj = None, X = None, S = None, c_const = None):
          
    
    if S is None:
          S = solvr_obj.S
          
    if X is None:
          X = solvr_obj.X
    if c_const is None:
    		c_const = solvr_obj.c_const
    
    Lambda = np.diag(lambda_vars)
    
    trace_mat1 = X.T @ X
    trace_mat2 = X @ S.T
    trace_mat3 = np.linalg.inv(S @ S.T + Lambda)
    trace_mat4 = (X @ S.T).T
    trace_mat5 = c_const * Lambda


    return -1 * (np.trace(trace_mat3at1) - np.trace(trace_mat2 @ trace_mat3 @ trace_mat4) - np.trace(trace_mat5))
    


def lagrange_dual_learn(solvr_obj = None, S = None, n = None, X = None, c_const = None, x0 = None, OptMethod = 'CG'):
    #B = solvr_obj.B
    #S = solvr_obj.S
    #n = solvr_obj.n
    
    if X is None:
        X = solvr_obj.X
    
    if S is None:
        S = solvr_obj.S
        
    if n is None:
        n = solvr_obj.n
	
    if c_const is None:
        c_const = solvr_obj.c_const

    #Initial guess = x0. If none, set to zeros (optimal for near optimal bases)
    if x0 is None:
        x0 = np.zeros(n)
    

    #Solve for optimal lambda
    lambda_vars = sopt.minimize(lagrange_dual_func, x0, method = OptMethod, args = (solvr_obj, X, S, c_const))
    #Set Lambda
    Lambda = np.diag(lambda_vars.x)
    
    #Returns B^T, for B corresponding to basis matrix
    #FEEL FREE TO TRANSPOSE IF YOU PREFER B BEING RETURNED INSTEAD
    BT = np.linalg.inv(S @ S.T + Lambda) @ (X @ S.T).T

    return BT
    


###### LAZY BOI TESTS ##################
#n0 = 6
#m0 = 5
#k0 = 3
#print(lagrange_dual_learn(S = np.random.randint(5, size = (n0,m0)), X = np.random.randint(5, size = (k0,m0)), n = n0, c_const = 0.00001))
#############################################
    
    
    
    
    
    
    

	# TODO: remove call to dummy solver
	#from dummy_solver import DUMMY_lagrange_dual
	#return DUMMY_lagrange_dual(solvr_obj, X, c_const)

	
	#FILLER, not permanent
	# return (X @ np.linalg.inv(S)).T





