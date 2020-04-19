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

		# TODO: initialize B properly?
		# self.B = np.full((k,n), np.nan, dtype = np.float)
		self.B = np.random.rand(k,n)
		
		# need to know training data to get dimension m
		self.m = None
		self.S = None

		# can also pass input data to ctor
		if X is not None:
			self.set_input(X, c_const, gamma)

	def print_cfg(self):
		print('\tn    \t:\t%d' % self.n)
		print('\tk    \t:\t%d' % self.k)
		print('\tm    \t:\t%d' % self.m)
		print('\tX dim\t:\t%s' % str(self.X.shape))
		print('\tB dim\t:\t%s' % str(self.B.shape))
		print('\tS dim\t:\t%s' % str(self.S.shape))
		# print('X data\t:\t%s' % str(self.X))

	def set_input(self, X, c_const = None, gamma = None):
		_, self.m = X.shape
		
		self.X = X
		
		# TODO: initialize S according to distribution
		# self.S = np.zeros((self.n, self.m), dtype = np.float)
		self.S = np.zeros((self.n, self.m), dtype = np.float)

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

	def train(self, delta : float, verbose = False):
		r'''
		X is input matrix, S is coefficient matrix,
		self.B is the basis matrix

		X \in \R^{k \times m}
		B \in \R^{k \times n}
		S \in \R^{n \times m}
		'''

		val = float('inf')
		val_new = self.value()
		rem = float('inf')

		iters = 0

		if verbose:
			print('\t  {:10s}   {:3s}'.format('iter', 'rem'))
			print('\t' + '-'*20)
			print('\t{:5d}\t{:10f}'.format(iters, rem))

		# TODO: not sure if this loop is right
		while not is_zero(rem, delta):
			for i in range(self.m):
				self.S[i] = feature_sign_search(self.B, self.X[:,i], self.gamma)
			self.B = lagrange_dual_learn(self)

			val = val_new
			val_new = self.value()
			rem = val - val_new

			if verbose:
				print('\t{:5d}\t{:10f}'.format(iters, rem))
			iters += 1


		return {
			'X' : self.X,
			'B' : self.B,
			'S' : self.S,
			'iters' : iters,
		}
