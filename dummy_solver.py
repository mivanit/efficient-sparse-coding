import numpy as np
import scipy.optimize as sopt

from util import *

# * dummy solvers to test portions of code, and have a baseline to compare to

def dummy_solver(solvr_obj, X, sigma, beta, c_const):
	r'''
	solve the whole darn problem using a standard solver
	X \in \R^{k \times m}
	'''
	m = X.shape[1]
	k = solvr_obj.k
	n = solvr_obj.n

	def mini_me(BS_arr):
		r'''
		function to minimize

		|| X - BS ||_F^2 / (2 * \sigma^2) + \beta \sum_{i,j} \rho(S_{i,j})
		'''
		B,S = np.split(BS_arr, k * m)
		np.reshape(B, (k, n))
		np.reshape(S, (k, m))

		return norm_F(X - (B @ S))**2 / (2 * sigma**2) + beta * sum(map(phi, S))


	def constraint_factory(j):
		r'''
		each column sum of squares should be less than `c_const`
		kind of funky because we want to create a separate constraint 
		for every j \in \N_n

		also, unpacking BS_arr every time is pointless 
		so we just slice the array fancily

		constraints are of the form:
		\sum_i B_{i,j}^2 \leq c, 	\forall j \in \N_n
		'''
		def constraint(BS_arr):				
			# # sum([ BS_arr[i*n + j]**2 for i in range(k) ])
			
			# gather the ith column by taking every nth element
			# starting at BS_arr[j] which is just B[0,j]
			B_col_i = BS_arr[j:(k * n):n]
			return c_const - sum([ v**2 for v in B_col_i ])
		
		return constraint
	
	# TODO: S_guess according to the given distribution
	S_guess = np.zeros((k * m))

	return sopt.minimize(
		fun = mini_me,
		x0 = S_guess,
		constraints = [ 
			{'type' : 'ineq', 'fun' : constraint_factory(j) } 
			for j in range(n)
		]
	)