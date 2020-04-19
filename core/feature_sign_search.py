'''
python implementation of the algorithms described in the paper
"Efficient sparse coding algorithms"
by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

for Math 651 at the University of Michigan, Winter 2020

by Michael Ivanitskiy and Vignesh Jagathese
'''

import numpy as np

from util import *

def feature_sign_search(A, y, gamma):
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

	y = y.reshape(*y.shape, 1)
	x = np.zeros(dim_p)

	## theta = lambda i: (x[i] > 0) - (x[i] < 0)
	## theta = np.array( x_i / abs(x_i) for x_i in x )
	theta = np.array(list(map(sign, x)))
	active_set = set()
	opmCond_a, opmCond_b = False, False

	# * 1.B: helpers
	def FS_unconstrained_QP_factory(M_hat, sgn_vec):
		'''
		objective for feature-sign step
		used only during line search from x_hat to x_hat_new
		'''
		return lambda x_vec : norm_2(y - M_hat @ x_vec)**2 + gamma * sgn_vec.T @ x_vec

	def deriv_yAx(i, x_i = 0.0):
		r'''
		returns the derivative
			\frac{\partial || y - Ax ||^2 }{ \partial x_i }
		at x_i = 0 by default
		note this is equalt to
			-2 * \sum_{j \in \N_m} 
			[ y_j - \sum_{k \in \N_p} A_{j,k} x_k ] 
			* [ A_{j,i} ]
		see paper_notes.md
		'''
		x_cpy = x
		# x_cpy = np.copy(x)
		# x_cpy[i] = x_i

		return (-2) * sum(
			A[j,i] * ( y[j] - sum(
				A[j,k] * x_cpy[k]
				for k in range(dim_p)
			) )
			for j in range(dim_m)
		)

	
	while not opmCond_b:

		# * 2: from zero coefficients of x, select i such that
		# 		y - A x is changing most rapidly with respect to x_i
		selector_arr = np.array([
			(
				# TODO: missing absolute value here?
				sum(A[i])
				if is_zero(x[i])
				else np.float('-inf')
			)
			for i in range(dim_p)
		])

		# REVIEW: if we pick an `i` that does not improve objective, try the next best `i`?

		looking_i = True
		num_tested_i = 0

		while looking_i:

			i_sel = np.argmax(np.absolute(selector_arr))

			# acivate x_i if it locally improves objective
			i_deriv = deriv_yAx(i_sel)
			if abs(i_deriv) > gamma:
				theta[i_sel] = (-1) * sign(i_deriv)
				active_set.add(i_sel)
				looking_i = False
			else:
				selector_arr[i_sel] = np.float('-inf')
				num_tested_i += 1
				if num_tested_i >= dim_p:
					print('no valid index in x_hat found, selecting at random')
					theta[i_sel] = (-1) * sign(i_deriv)
					active_set.add(i_sel)

		active_list = sorted(list(active_set))
	
		while not opmCond_a:

			# * 3: feature-sign step

			# A_hat is submatrix of A containing only columns corresponding to active set
			## A_hat = A[:,active_list]
			## x_hat = x[active_list]
			## theta_hat = np.array([sign(x[a]) for a in active_list])
			A_hat = select_cols(A, active_list)
			x_hat = select_elts(x, active_list)
			theta_hat = np.array([sign(a) for a in x_hat])

			# compute solution to unconstrained QP:
			# minimize_{x_hat} || y - A_hat @ x_hat ||^2 + gamma * theta_hat.T @ x_hat

			# REVIEW: do we /really/ need to compute matrix inverse? can we minimize or at least compute inverse more efficiently?

			print(A_hat.shape)
			print(A_hat.T.shape)
			print(y.shape)
			
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

			# update x to x_hat
			for idx_activ in range(len(active_list)):
				x[active_list[idx_activ]] = x_hat[idx_activ]

			# remove zero coefficients of x_hat from active set, update theta
			for idx_rem in range(len(x_hat)):
				if is_zero(x_hat[idx_rem]):
					active_set.remove(idx_rem)
					A_hat, x_hat, theta_hat, x_hat_new = (None for _ in range(4))
			
			theta = np.array(list(map(sign, x)))

			# * 4: check optimality condition a
			
			# set of j where x_j != 0
			set_j_not0 = {
				j for j in range(dim_p)
				if not is_zero( x[j] )
			}

			opmCond_a = all( 
				is_zero( deriv_yAx(j) + gamma * sign(x[j]) )
				for j in set_j_not0
			)



		# * 4: check optimality condition b
		# set of j where x_j == 0
		set_j_is0 = {
			j for j in range(dim_p)
			if is_zero( x[j] )
		}

		opmCond_b = all( 
			abs(deriv_yAx(j)) <= gamma
			for j in set_j_is0
		)

	return x

