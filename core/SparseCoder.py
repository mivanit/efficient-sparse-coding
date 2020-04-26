import numpy as np

from util import *

import feature_sign_search as fss
import lagrange_dual_learn as ldl
import dummy_solver as dum


DEFAULT = {
	'c_const'	: 1e-2,
	'sigma'		: 1,
	'beta'		: 1,
}


class SparseCoder(object):

	def __init__(self,
		n, k, X = None,
		c_const = None, sigma = None, beta = None,
	):
		'''
		n is dimension of basis,
		k is data dimension
		'''
		
		self.n = int(n)
		self.k = int(k)

		# TODO: initialize B properly?
		# self.B = np.full((k,n), np.nan, dtype = np.float)
		self.B = np.random.rand(k,n) * c_const
		
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
		print('\tc_const\t:\t%f' % self.c_const)
		print('\tsigma  \t:\t%f' % self.sigma)
		print('\tbeta   \t:\t%f' % self.beta)
		print('\tgamma  \t:\t%f' % self.gamma)
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

		# compute gamma for feature sign search
		self.gamma = 2 * (self.sigma**2) * self.beta

		self.print_cfg()		

	def value(self):
		# print(self.X.shape)
		# print(self.B.shape)
		# print(self.S.shape)

		return (
			norm_F(self.X - (self.B @ self.S))**2.0 
			+ 2 * (self.sigma**2.0) * self.beta * sum(map(phi, self.S)) 
		)

	def train(self, delta : float, verbose = False,
		max_iter = float('inf'), method = 'std',
	):
		r'''
		X is input matrix, S is coefficient matrix,
		self.B is the basis matrix

		X \in \R^{k \times m}
		B \in \R^{k \times n}
		S \in \R^{n \times m}

		methods:
			`std` 	  : standard, non-dummy solver
			`ds_lag`  : langrange dual using dummy solver
			`ds_feat` : feature sign search using dummy solver
			`ds_comb` : both lagrange dual and feature sign using dummy solver
		'''

		# show settings
		print('\tmax_iter\t:\t%f' % max_iter)
		print('\tmethod  \t:\t%s' % method)
		print('\tdelta   \t:\t%s' % delta)

		# select method
		if method in ('ds_feat','ds_comb'):
			feature_sign_search = dum.feature_sign_search
		else:
			feature_sign_search = fss.feature_sign_search
		
		if method in ('ds_lag','ds_comb'):
			lagrange_dual_learn = dum.lagrange_dual_learn
		else:
			lagrange_dual_learn = ldl.lagrange_dual_learn

		# inititalize values
		val = [float('inf') for i in range(4)]
		self.val_data = [val]
		iters = 0

		# print verbose header
		print('='*70)
		if verbose:
			print('%15s %15s %15s %15s %15s' % ('iter','val','val_lag','val_new','rem'))
			print('\t' + '-'*70)
			print('%15s %15f %15f %15f %15f' % (iters, *val))


		Lambda = np.zeros(self.n)

		# iterate until within delta of 0
		while (not is_zero(val[3], delta)) and (iters < max_iter):

			temp_prev_val = val[2]
			if iters > 3:
				rand_mod = delta * val[3]
			else:
				rand_mod = delta
			val = [None for i in range(4)]
			val[0] = temp_prev_val

			# val[0] : old value
			# val[1] : first step (lagrange)
			# val[2] : new value (both steps)
			# val[3] : remainder (new - old)

			# lagrange step
			self.B, Lambda = lagrange_dual_learn(
				X = self.X,
				S = self.S,
				n = self.n,
				# L_init = Lambda, # TODO: does this help or no?
				c_const = self.c_const,
			)

			val[1] = self.value()

			# feature sign step
			for i in range(self.m):
				self.S[:,i] = feature_sign_search(
					self.B, 
					self.X[:,i], 
					self.gamma, 
					x0 = vec_randomize(self.S[:,i], rand_mod), # TODO: does this help or no?
				)

			val[2] = self.value()
			val[3] = val[2] - val[0]

			self.val_data.append(val.copy())

			if verbose:
				print('%15s %15f %15f %15f %15f' % (iters, *val))
			iters += 1

		# return results dict
		return {
			'X' : self.X,
			'B' : self.B,
			'S' : self.S,
			'iters' : iters,
			'val'	: np.array(self.val_data),
		}


