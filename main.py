import numpy as np

# norm shorthands
norm_1 = lambda x : np.linalg.norm(x, 1)




class SparseCode(object):

	def __init__(self, n, k):
		'''
		n is dimension of basis,
		k is data dimension
		'''
		
		self.n = n
		self.k = k
		self.B = np.full((n,k), np.nan)

	def train(X):
		r'''
		X is input matrix, S is coefficient matrix,
		self.B is the basis matrix

		X,S \in \R^{k \times m}
		B \in \R^{k \times n}
		'''


	def represent(self, xi):
		'''
		given input vector `xi` \in \R^k, 
		return a vector `s` \in \R^n such that
		xi ~= \sum_j b[j] s[j]
		
		to favor sparse coefficients:
		prior distribution of each s[j] is defined as
		'''

		# TODO









def feature_sign_search(A, y, N = None):
	'''
	inputs: matrix A, dimension N
	''' 

	x = np.zeros(N)

	theta = lambda i: (x[i] > 0) - (x[i] < 0)

	active_set = set()




