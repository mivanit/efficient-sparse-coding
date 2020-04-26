import numpy as np
import scipy.optimize as sopt

#from util import *

# Lagrange Dual Function to be maximized w/r/t lambda_vars
# Returns negative value because we want to maximize it using a minimization function

Tr = np.trace
inv = np.linalg.inv

def lagrange_dual_factory(X, S, c_const):
    
    def mini_me(lambda_vars):
        Lambda = np.diag(lambda_vars)

        return (
            -1 * Tr(X.T @ X) 
            - Tr((X @ S.T) @ (inv(S @ S.T + Lambda)) @ (X @ S.T).T) 
            - Tr(c_const * Lambda)
        )

    return mini_me
    


def lagrange_dual_learn(X, S, n, c_const, B_init = None, method = 'CG'):

    # Initial guess = x0. If none, set to zeros (optimal for near optimal bases)
    if B_init is None:
        B_init = np.zeros(n)

    # Solve for optimal lambda
    lambda_vars = sopt.minimize(
        lagrange_dual_factory(X,S,c_const),
        B_init, 
        method = method,
    )

    # Set Lambda
    Lambda = np.diag(lambda_vars.x)

    # Returns B^T, for B corresponding to basis matrix
    B = (np.linalg.inv(S @ S.T + Lambda) @ (X @ S.T).T).T

    return B
    






# * LAZY BOI TESTS
# n0 = 60
# m0 = 50
# k0 = 30
# print(lagrange_dual_learn(S = np.random.randint(5, size = (n0,m0)), X = np.random.randint(5, size = (k0,m0)), n = n0, c_const = 0.001))

    
    
    
# * FILLER, not permanent
# return (X @ np.linalg.inv(S)).T


# * old code:
# def mini_me(lambda_vars):
#         Lambda = np.diag(lambda_vars)
        
# trace_mat1 = X.T @ X
# trace_mat2 = X @ S.T
# trace_mat3 = np.linalg.inv(S @ S.T + Lambda)
# trace_mat4 = (X @ S.T).T
# trace_mat5 = c_const * Lambda

# return -1 * np.trace(trace_mat1) - np.trace(trace_mat2 @ trace_mat3 @ trace_mat4) - np.trace(trace_mat5)




