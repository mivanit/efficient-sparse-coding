import numpy as np

from util import *

def lagrange_dual_learn(solvr_obj, X, c_const):
	B = solvr_obj.B
	S = solvr_obj.S
	n = solvr_obj.n