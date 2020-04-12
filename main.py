import numpy as np

import scipy.optimize as sopt

# norm shorthands
norm_1 = lambda x : np.linalg.norm(x, 1)
norm_F = lambda x : np.linalg.norm(x, 'fro')

# penalty function \phi
phi = lambda x : norm_1(x)



class SparseCode(object):

	def __init__(self, n, k):
		'''
		n is dimension of basis,
		k is data dimension
		'''
		
		self.n = n
		self.k = k
		self.B = np.full((n,k), np.nan)


	def dummy_solver(self, X, sigma, beta, c_const):
		r'''
		X \in \R^{k \times m}
		'''
		m = X.shape[1]

		def mini_me(BS_arr):
			'''
			function to minimize
			'''
			B,S = np.split(BS_arr, self.k * m)
			np.reshape(B, (self.k, self.n))
			np.reshape(S, (self.k, m))

			return norm_F(X - (B @ S))**2 / (2 * sigma**2) + beta * sum(map(phi, S))


		def constraint_factory(j):
			r'''
			each column sum of squares should be less than `c_const`
			kind of funky because we want to create a separate constraint 
			for every j \in \N_n

			also, unpacking BS_arr every time is pointless 
			so we just slice the array fancily
			'''
			def constraint(BS_arr):				
				# # sum([ BS_arr[i*self.n + j]**2 for i in range(self.k) ])
				
				# gather the ith column by taking every nth element
				# starting at BS_arr[j] which is just B[0,j]
				B_col_i = BS_arr[j:(self.k * self.n):self.n]
				return c_const - sum([ v**2 for v in B_col_i ])
			
			return constraint
		
		# TODO: S_guess
		S_guess = np.zeros((self.k * m))

		sopt.minimize(
			fun = mini_me,
			x0 = S_guess,
			constraints = [ 
				{'type' : 'ineq', 'fun' : constraint_factory(j) } 
				for j in range(self.n)
			]
		)

			


			


	def train(self, X):
		r'''
		X is input matrix, S is coefficient matrix,
		self.B is the basis matrix

		X,S \in \R^{k \times m}
		B \in \R^{k \times n}
		'''
		# TODO
		pass


	def represent(self, xi):
		'''
		given input vector `xi` \in \R^k, 
		return a vector `s` \in \R^n such that
		xi ~= \sum_j b[j] s[j]
		
		to favor sparse coefficients:
		prior distribution of each s[j] is defined as
		'''
		# TODO
		pass





def feature_sign_search(A, y, N = None):
	'''
	inputs: matrix A, dimension N
	''' 

	# 1: initialize

	x = np.zeros(N)

	theta = lambda i: (x[i] > 0) - (x[i] < 0)

	active_set = set()

	# 2: from zero coefficients of x, select i = argmax(stuff)
	selector = [
		(
			abs(None) #TODO: this derivative is rally easy and literally linear
			if x[i] == 0
			else None
		)
		for i in range(N)
	]




