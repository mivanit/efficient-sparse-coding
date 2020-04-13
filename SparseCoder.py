class SparseCoder(object):

	def __init__(self, n, k):
		'''
		n is dimension of basis,
		k is data dimension
		'''
		
		self.n = n
		self.k = k
		self.B = np.full((n,k), np.nan)

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

