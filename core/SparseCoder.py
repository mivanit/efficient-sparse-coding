import numpy as np

from util import *

from feature_sign_search import feature_sign_search
from lagrange_dual_learn import lagrange_dual_learn

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
		
		self.n = int(n)
		self.k = int(k)
		self.B = np.full((k,n), np.nan, dtype = np.float)
		
		# need to know training data to get dimension m
		self.m = None
		self.S = None

		# can also pass input data to ctor
		if X is not None:
			self.set_input(X, c_const, gamma)

	def print_cfg(self):
		print('n    \t:\t%d' % self.n)
		print('k    \t:\t%d' % self.k)
		print('k    \t:\t%d' % self.m)
		print('X dim\t:\t%s' % self.X.shape)
		print('B dim\t:\t%s' % self.B.shape)
		print('S dim\t:\t%s' % self.S.shape)

	def set_input(self, X, c_const = None, gamma = None):
		_, self.m = X.shape
		
		self.X = X
		
		# TODO: initialize S according to distribution
		self.S = np.zeros((self.n, self.m), 0.0)

		self.print_cfg()

		
		
		if c_const is not None:
			self.c_const = float(c_const)
		else:
			self.c_const = DEFAULT['c_const']

		if gamma is not None:
			self.gamma = float(gamma)
		else:
			self.gamma = DEFAULT['gamma']


	def value(self):
		return (
			norm_F(self.X - (self.B @ self.S))**2 
			+ self.gamma * sum(map(phi, self.S)) 
		)

	def train(self, delta : float):
		r'''
		X is input matrix, S is coefficient matrix,
		self.B is the basis matrix

		X \in \R^{k \times m}
		B \in \R^{k \times n}
		S \in \R^{n \times m}
		'''

		val = float('inf')
		val_new = self.value()

		iters = 0

		# TODO: not sure if this loop is right
		while is_zero(val - val_new, delta):
			for i in range(self.m):
				self.S[:,i] = feature_sign_search(self.B, self.X[:,i], self.gamma)
			self.B = lagrange_dual_learn(self)
			
			iters += 1


		return {
			'X' : self.X,
			'B' : self.B,
			'S' : self.S,
			'iters' : iters,
		}
