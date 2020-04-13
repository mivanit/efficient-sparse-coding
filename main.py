import numpy as np

import scipy.optimize as sopt


##     ## ######## #### ##
##     ##    ##     ##  ##
##     ##    ##     ##  ##
##     ##    ##     ##  ##
##     ##    ##     ##  ##
##     ##    ##     ##  ##
 #######     ##    #### ########


# sign function
sign = lambda x : int(x / abs(x)) if x != 0 else 0

# norm shorthands
norm_1 = lambda x : np.linalg.norm(x, 1)
norm_F = lambda x : np.linalg.norm(x, 'fro')

# penalty function \phi
phi = lambda x : norm_1(x)


# float comparison threshold
THRESH = 1e-7

def is_zero(x):
	return abs(x) < THRESH


#### ##    ## #### ########
 ##  ###   ##  ##     ##
 ##  ####  ##  ##     ##
 ##  ## ## ##  ##     ##
 ##  ##  ####  ##     ##
 ##  ##   ###  ##     ##
#### ##    ## ####    ##

class SparseCode(object):

	def __init__(self, n, k):
		'''
		n is dimension of basis,
		k is data dimension
		'''
		
		self.n = n
		self.k = k
		self.B = np.full((n,k), np.nan)

	########  ##     ## ##     ## ##     ## ##    ##
	##     ## ##     ## ###   ### ###   ###  ##  ##
	##     ## ##     ## #### #### #### ####   ####
	##     ## ##     ## ## ### ## ## ### ##    ##
	##     ## ##     ## ##     ## ##     ##    ##
	##     ## ##     ## ##     ## ##     ##    ##
	########   #######  ##     ## ##     ##    ##

	# * dummy solvers to test portions of code, and have a baseline to compare to

	def dummy_solver(self, X, sigma, beta, c_const):
		r'''
		solve the whole darn problem using a standard solver
		X \in \R^{k \times m}
		'''
		m = X.shape[1]

		def mini_me(BS_arr):
			r'''
			function to minimize

			|| X - BS ||_F^2 / (2 * \sigma^2) + \beta \sum_{i,j} \rho(S_{i,j})
			'''
			B,S = np.split(BS_arr, self.k * m)
			np.reshape(B, (self.k, self.n))
			np.reshape(S, (self.k, m))

			return norm_F(X - (B @ S))**2 / (2 * sigma**2) + beta * sum(map(phi, S))


		def constraint_factory(j):
			r'''
			each column sum of squares should be less than `c_const`
			kind of funky because we want to create a separate constraint 
			for every j \in \N_n

			also, unpacking BS_arr every time is pointless 
			so we just slice the array fancily

			constraints are of the form:
			\sum_i B_{i,j}^2 \leq c, 	\forall j \in \N_n
			'''
			def constraint(BS_arr):				
				# # sum([ BS_arr[i*self.n + j]**2 for i in range(self.k) ])
				
				# gather the ith column by taking every nth element
				# starting at BS_arr[j] which is just B[0,j]
				B_col_i = BS_arr[j:(self.k * self.n):self.n]
				return c_const - sum([ v**2 for v in B_col_i ])
			
			return constraint
		
		# TODO: S_guess according to the given distribution
		S_guess = np.zeros((self.k * m))

		return sopt.minimize(
			fun = mini_me,
			x0 = S_guess,
			constraints = [ 
				{'type' : 'ineq', 'fun' : constraint_factory(j) } 
				for j in range(self.n)
			]
		)


    ######   #######  ##       ##     ## ######## ########
   ##    ## ##     ## ##       ##     ## ##       ##     ##
   ##       ##     ## ##       ##     ## ##       ##     ##
    ######  ##     ## ##       ##     ## ######   ########
         ## ##     ## ##        ##   ##  ##       ##   ##
   ##    ## ##     ## ##         ## ##   ##       ##    ##
    ######   #######  ########    ###    ######## ##     ##

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






   ###    ##        ######    #######   ######
  ## ##   ##       ##    ##  ##     ## ##    ##
 ##   ##  ##       ##        ##     ## ##
##     ## ##       ##   #### ##     ##  ######
######### ##       ##    ##  ##     ##       ##
##     ## ##       ##    ##  ##     ## ##    ##
##     ## ########  ######    #######   ######



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











