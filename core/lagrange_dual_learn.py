import numpy as np

from util import *

def lagrange_dual_learn(solvr_obj, X, c_const):
    B = solvr_obj.B
    S = solvr_obj.S
    n = solvr_obj.n

    # TODO: remove call to dummy solver
    from dummy_solver import DUMMY_lagrange_dual
    return DUMMY_lagrange_dual(solvr_obj, X, c_const)

    
    #FILLER, not permanent
    # return (X @ np.linalg.inv(S)).T