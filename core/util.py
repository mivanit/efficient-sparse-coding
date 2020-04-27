import numpy as np

# norm shorthands
norm_1 = lambda x : np.linalg.norm(x, 1)
norm_2 = lambda x : np.linalg.norm(x, 2)
norm_F = lambda x : np.linalg.norm(x, 'fro')

# penalty function \phi
phi = lambda x : norm_1(x)


# float comparison threshold
THRESH = 1e-10

def is_zero(x, thresh = THRESH):
	return abs(x) < thresh

# sign function
sign = np.sign

# sign = lambda x : int(x / abs(x)) if not is_zero(x) else 0
# def sign_v(vec):
# 	return np.array([ sign(v) for v in vec ])



def argmin(arr):
	min_idx = 0
	min_val = arr[0]

	for i in range(len(arr)):
		if arr[i] < min_val:
			min_val = arr[i]
			min_idx = i
	
	return min_idx



def argmin_f(arr, func):
	'''
	gives the index to `arr` such that `func(arr[i])` is minimized
	'''
	min_idx = 0
	min_val = func(arr[0])

	for i in range(len(arr)):
		temp = func(arr[i])
		if temp < min_val:
			min_val = temp
			min_idx = i
	
	return min_idx
	




def coeff_discrete_line_search(u, v, func):
	'''
	check every point w on the closed line from u to v
	(including endpoints) where a coefficient changes sign
	to minimize func(w)
	'''
	# direction vector
	x = v - u

	# vectors to check
	W = [u, v]

	# REVIEW: probably can do this w/ generators and better vectorization
	
	# if sign change happens, compute where
	sign_change = [ 
		(
			# v + c * x should be 0 at index i so:
			# c = - v[i] / x[i]
			x + ( - v[i] / x[i] ) * x
			if not (sign(u[i]) == sign(v[i])) 
			else None
		)
		for i in range(len(u))
	]

	# add vectors to check if sign changes, find min
	W = W + [ a for a in sign_change if a is not None ]

	# find min
	return W[argmin_f(W, func)]




def select_elts(arr, lst_idx):
	return np.array([arr[i] for i in lst_idx])

def select_rows(arr, lst_rows):
	if len(lst_rows) > 1:
		row_data = (arr[i] for i in lst_rows)
		return np.concatenate(row_data,1)
	elif len(lst_rows) == 1:
		return np.array([arr[lst_rows[0]]])
	else:
		raise Exception('select_rows: lst_rows empty')

def select_cols(arr, lst_cols):
	if len(lst_cols) > 1:
		col_data = (arr[i] for i in lst_cols)
		return np.concatenate(col_data,0)
	elif len(lst_cols) == 1:
		return np.array([arr[lst_cols[0]]]).T
	else:
		raise Exception('select_cols: lst_cols empty')



def inv(arr):
	'''
	computes inverse, allowing for singular matricies
	'''
	# print(arr.shape)
	if arr.shape == (1,1):
		return np.array([[ 1/arr[0][0] ]])
	else:
		return np.linalg.inv(arr)


def vec_randomize(v, s):
	'''
	modifies each element of the vector v[i] by some coefficient q[i]
	where q[i] ~ Normal(1, s)
	'''
	return v * np.random.normal(1.0, abs(s), v.shape)



# for calculating optimal h when numerically taking derivative
# see: https://en.wikipedia.org/wiki/Numerical_differentiation#Step_size
MACHINE_EPSILON = 2.22e-16
ROOT_MACHINE_EPSILON = 1.48997-8

def numerical_derivative(f, x, comp = None, verb = False):
	'''
	uses symmetric difference quotient
	'''
	h = max(
		ROOT_MACHINE_EPSILON * x, 
		MACHINE_EPSILON
	)

	if verb:
		print('\t%f\t%f\t%f\t%f' % (h, f(x-h), f(x), f(x+h)))

		import matplotlib.pyplot as plt
		h = h*0.1
		x_arr = [ (x + a*h*0.05) for a in range(-1000,1000)]
		f_arr = [ f(x_) for x_ in x_arr ]
		plt.plot(x, f(x), 'ro')
		plt.plot(x+h, f(x+h), 'go')
		plt.plot(x-h, f(x-h), 'go')
		plt.plot(x_arr, f_arr,'b.-')
		if comp is not None:
			plt.plot([x-h, x+h], [f(x)-comp*h,f(x)+comp*h],'k-')
		plt.show()

	return ( f(x+h) - f(x-h) ) / (2 * h)

