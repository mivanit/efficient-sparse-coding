import numpy as np

from util import *

def lagrange_dual_learn(solvr_obj, X = None, c_const = None):
	B = solvr_obj.B
	S = solvr_obj.S
	n = solvr_obj.n

	if X is None:
		X = solvr_obj.X
	
	if c_const is None:
		c_const = solvr_obj.c_const

	# TODO: remove call to dummy solver
	from dummy_solver import DUMMY_lagrange_dual
	return DUMMY_lagrange_dual(solvr_obj, X, c_const)

	
	#FILLER, not permanent
	# return (X @ np.linalg.inv(S)).T