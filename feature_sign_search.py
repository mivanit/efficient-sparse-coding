'''
python implementation of the algorithms described in the paper
"Efficient sparse coding algorithms"
by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

for Math 651 at the University of Michigan, Winter 2020

by Michael Ivanitskiy and Vignesh Jagathese
'''

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

	x = np.zeros(dim_p)

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
		x_cpy = np.copy(x)
		x_cpy[i] = x_i

		return (-2) * sum(
			A[j,i] * ( y[j] - sum(
				A[j,k] * x_cpy[k]
				for k in range(dim_p)
			) )
			for j in range(dim_m)
		)


	## theta = lambda i: (x[i] > 0) - (x[i] < 0)
	## theta = np.array( x_i / abs(x_i) for x_i in x )
	theta = np.array(map(sign, x))

	active_set = set()

	opmCond_a, opmCond_b = False, False
	
	while not opmCond_b:

		# * 2: from zero coefficients of x, select i such that
		# 		y - A x is changing most rapidly with respect to x_i
		selector_arr = np.array([
			(
				sum(A[i])
				if x[i] == 0
				else np.float('-inf')
			)
			for i in range(dim_p)
		])

		i_sel = np.argmax(np.absolute(selector_arr))

		# acivate x_i if it locally improves objective
		i_deriv = deriv_yAx(i_sel)
		if abs(i_deriv) > gamma:
			theta[i_sel] = (-1) * sign(i_deriv)
			active_set.add(i_sel)

		active_list = sorted(list(active_set))
		
		while not opmCond_a:

			# * 3: feature-sign step

			# A_hat is submatrix of A containing only columns corresponding to active set
			# REVIEW: not sure if this line will work
			A_hat = A[:,active_list]

			# TODO: whats a clean way to handle x_hat indecies here?

			# compute solution to unconstrained QP:
			# minimize_{x_hat} || y - A_hat @ x_hat ||^2 + gamma * theta_hat.T @ x_hat

			# TODO: do we /really/ need to compute matrix inverse? can we minimize more efficiently?
			theta_hat = [sign(x[a]) for a in active_list]

			x_hat_new = np.linalg.inv(A_hat.T @ A_hat) @ (A_hat.T @ y - gamma * theta_hat / 2)

			# perform a discrete line search on segment x_hat to x_hat_new

			# TODO: this whole bit


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

