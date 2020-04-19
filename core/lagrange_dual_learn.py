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
    trace_mat = X.T @ X - X @ S.T @ (np.linalg.inv(S @ S.T + Lambda)) @ (X @ S.T).T - c_const * Lambda
    
    
    
    return -1 * np.trace(trace_mat) 
    


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
    lambda_vars = sopt.minimize(lagrange_dual_func, x0, method = OptMethod, args = (solvr_obj, S, X, c_const))
    
    #Set Lambda
    Lambda = np.diag(lambda_vars)
    
    #Returns B^T, for B corresponding to basis matrix
    #FEEL FREE TO TRANSPOSE IF YOU PREFER B BEING RETURNED INSTEAD
    BT = np.linalg.inv(S @ S.T + Lambda) @ (X @ S.T).T
    
    
    return BT
    


#print(lagrange_dual_func(lambda_vars = np.eye(5), X = np.eye(5), S = np.eye(5), c_const = 0.00001))
lagrange_dual_learn(S = np.random.randint(5, size = (6,5)), X = np.random.randint(5, size = (3,5)), n = 6, c_const = 0.001) 
    
    
    
    
    
    
    
    

	# TODO: remove call to dummy solver
	#from dummy_solver import DUMMY_lagrange_dual
	#return DUMMY_lagrange_dual(solvr_obj, X, c_const)

	
	#FILLER, not permanent
	# return (X @ np.linalg.inv(S)).T





