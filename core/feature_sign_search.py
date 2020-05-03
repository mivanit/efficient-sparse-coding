'''
python implementation of the algorithms described in the paper
"Efficient sparse coding algorithms"
by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

for Math 651 at the University of Michigan, Winter 2020

by Michael Ivanitskiy and Vignesh Jagathese
'''

import numpy as np

from util import *

def feature_sign_search(A, y, gamma, x0 = None):
	r'''
	inputs: 
		- matrix A \in \R^{m * p}
		- vector y \in \R^m

	note:
		x \in \R^p

		y - Ax = 
		[ y[j] - \sum_k A[j,k] x[k] ]_{j \in \N_m}
	
	see `paper_notes.md`. 
		\argmax_i | \frac{\partial || y - Ax ||^2 }{ \partial x_i } |
	reduces to
		\argmax_i | \sum\limits_{j \in \N_m} A_{j,i} |
	which is just the largest row sum
	''' 

	# * 1: initialize
	dim_m, dim_p = A.shape

	# reshapes y so can subtract Ax properly
	y = np.reshape(y, (-1,1)) 

	x = np.zeros((dim_p,1))
	if x0 is not None:
		x = x0

	theta = np.zeros((dim_p,1), dtype=np.int8)
	active_set = set()

	# * 1.B: helpers
	def FS_unconstrained_QP_factory(M_hat, sgn_vec):
		'''
		objective for feature-sign step
		used only during line search from x_hat to x_hat_new
		'''
		return lambda x_vec : norm_2(y - M_hat @ x_vec)**2 + gamma * sgn_vec.T @ x_vec


	# since x is all zeroes
	grad = - 2 * A.T @ y

	# gradient computation function
	def comp_grad():
		r'''
		computing the gradient:
		partial wrt x_i || y - Ax ||^2
		'''
		return -2 * A.T @ ( y - A @ x )
	
	opcond_b = np.inf
	opcond_a = 0

	while opcond_b > gamma:

		# * 2: select i
		# from zero coefficients of x, select i such that
		# y - A x is changing most rapidly with respect to x_i
		
		# acivate x_i if it locally improves objective			
		i_sel = np.argmax(np.abs(grad) * (theta == 0))
		i_deriv = grad[i_sel]
		
		# print('SELECTING:\t%d\t%f' % (i_sel, i_deriv))

		if i_deriv > gamma:
			theta[i_sel] = -1
		elif i_deriv < - gamma:
			theta[i_sel] = 1
		else:
			raise Exception('i_deriv too small mag!')
		
		x[i_sel] = 0.0
		active_set.add(i_sel)

		opcond_a = np.max(abs(grad[theta == 0]))
	
		while not np.allclose(opcond_a, 0):

			active_list = np.array(sorted(list(active_set)))
			print(active_list)

			# * 3: feature-sign step

			# A_hat is submatrix of A containing only columns corresponding to active set
			## A_hat = A[:,active_list]
			## x_hat = x[active_list]
			## theta_hat = np.array([sign(x[a]) for a in active_list])
			## theta_hat = np.array([sign(a) for a in x_hat])
			## A_hat = select_cols(A, active_list)
			A_hat = np.delete(A, [i for i in range(dim_p) if i not in active_set], 1)
			x_hat = select_elts(x, active_list)
			theta_hat = select_elts(theta, active_list)

			# compute solution to unconstrained QP:
			# minimize_{x_hat} || y - A_hat @ x_hat ||^2 + gamma * theta_hat.T @ x_hat

			# REVIEW: do we /really/ need to compute matrix inverse? can we minimize or at least compute inverse more efficiently?

			# print(theta_hat.shape)
			# print((A_hat.T @ y).shape)

			x_hat_new = (
				np.linalg.inv(A_hat.T @ A_hat)
				@
				(
					A_hat.T @ y
					- gamma * theta_hat / 2
				)
			)

			# perform a discrete line search on segment x_hat to x_hat_new

			line_search_func = FS_unconstrained_QP_factory(A_hat, theta_hat)

			x_hat = coeff_discrete_line_search(x_hat, x_hat_new, line_search_func)

			print(x_hat.T)

			# update x to x_hat
			print('updating x to x_hat')
			print(x.T)
			for idx_activ in range(len(active_list)):
				x[active_list[idx_activ]] = x_hat[idx_activ]
			print(x.T)
			print('-'*40)

			# remove zero coefficients of x_hat from active set, update theta
			for idx_rem in range(len(x_hat)):
				if is_zero(x_hat[idx_rem]):
					active_set.discard(idx_rem)
					A_hat, x_hat, theta_hat, x_hat_new = (None for _ in range(4))
			
			theta = sign(x)

			grad = comp_grad()

			opcond_a = np.max(abs(grad[theta == 0]))

		grad = comp_grad()

		opcond_b = np.max(abs(grad[theta != 0] + gamma * theta[theta != 0]))

		
	def opCond_a():
		# * 4: check optimality condition a
		# set of j where x_j != 0

		print(x.T)
		set_j_not0 = {
			j for j in range(dim_p)
			if not is_zero( x[j] )
		}

		print('-'*50)
		print('\tA:\t%s' % str(set_j_not0))
		for j in set_j_not0:
			# print('\t' + str(deriv_yAx(j)))
			print('\t' + str(deriv_yAx(j) + gamma * sign(x[j])))


		return all( 
			# is_zero( deriv_yAx(j) + gamma * sign(x[j]) )
			( deriv_yAx(j) + gamma * sign(x[j]) ) == 0
			for j in set_j_not0
		) and set_j_not0


	def opCond_b():
		# * 4: check optimality condition b
		# set of j where x_j == 0
		
		print(x.T)
		set_j_is0 = {
			j for j in range(dim_p)
			if is_zero(x[j])
		}
		# REVIEW: is strict equality ok here?


		print('*'*50)
		print('\tB:\t%s' % str(set_j_is0))
		for j in set_j_is0:
			print('\t' + str(deriv_yAx(j)))

		return all( 
			abs(deriv_yAx(j)) <= gamma
			for j in set_j_is0
		)


	return x

