import numpy as np

from util import *

from feature_sign_search import feature_sign_search
from lagrange_dual_learn import lagrange_dual_learn

DEFAULT = {
	'c_const'	: 1e-2,
	'sigma'		: 1,
	'beta'		: 1,
}


class SparseCoder(object):

	def __init__(self, n, k, X = None, c_const = None, sigma = None, beta = None):
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
			self.set_input(X, c_const, sigma, beta)

	def print_cfg(self):
		print('\tn    \t:\t%d' % self.n)
		print('\tk    \t:\t%d' % self.k)
		print('\tm    \t:\t%d' % self.m)
		print('\tX dim\t:\t%s' % str(self.X.shape))
		print('\tB dim\t:\t%s' % str(self.B.shape))
		print('\tS dim\t:\t%s' % str(self.S.shape))
		# print('X data\t:\t%s' % str(self.X))

	def set_input(self, X, c_const = None, sigma = None, beta = None):
		_, self.m = X.shape
		
		self.X = X

		# init S randomly
		self.S = np.random.random((self.n, self.m))
		# get exponenitially distributed scaling factors
		temp_S_mags = np.random.exponential(beta, self.m)
		# scale each vector such that the norm is as generated
		temp_S_scale = np.array([ temp_S_mags[i] / norm_1(self.S[:,i]) for i in range(self.m) ])
		self.S = self.S * temp_S_scale

		self.print_cfg()
		
		if c_const is not None:
			self.c_const = float(c_const)
		else:
			self.c_const = DEFAULT['c_const']

		if sigma is not None:
			self.sigma = float(sigma)
		else:
			self.sigma = DEFAULT['sigma']

		if beta is not None:
			self.beta = float(beta)
		else:
			self.beta = DEFAULT['beta']


	def value(self):
		return (
			norm_F(self.X - (self.B @ self.S))**2.0 
			+ 2 * (self.sigma**2.0) * self.beta * sum(map(phi, self.S)) 
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

		# compute gamma for feature sign search
		gamma = 2 * (self.sigma**2) * self.beta

		# iterate until within delta of 0
		while not is_zero(rem, delta):

			# lagrange step
			self.B = lagrange_dual_learn(self)

			# feature sign step
			for i in range(self.m):
				self.S[i] = feature_sign_search(self.B, self.X[:,i], gamma)

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
