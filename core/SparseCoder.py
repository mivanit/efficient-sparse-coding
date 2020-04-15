import numpy as np


DEFAULT = {
	'c_const'	: 1e-2,
	'gamma'		: 1e-5,
}


class SparseCoder(object):

	def __init__(self, n, k, X = None, c_const = None, gamma = None):
		'''
		n is dimension of basis,
		k is data dimension
		'''
		
		self.n = n
		self.k = k
		self.B = np.full((k,n), np.nan, dtype = np.float)
		
		# need to know training data to get dimension m
		self.m = None
		self.S = None

		# can also pass input data to ctor
		if X is not None:
			self.set_input(X, c_const, gamma)


	def set_input(self, X, c_const = None, gamma = None):
		_, self.m = X.shape
		
		# TODO: initialize S according to distribution
		self.S = np.zeros((self.k, self.m), 0.0)
		
		if c_const is not None:
			self.c_const = c_const
		else:
			self.c_const = DEFAULT['c_const']

		if gamma is not None:
			self.gamma = gamma
		else:
			self.gamma = DEFAULT['gamma']



	def train(self, X):
		r'''
		X is input matrix, S is coefficient matrix,
		self.B is the basis matrix

		X,S \in \R^{k \times m}
		B \in \R^{k \times n}
		'''
		# TODO
		pass
