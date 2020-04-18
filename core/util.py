import numpy as np

# norm shorthands
norm_1 = lambda x : np.linalg.norm(x, 1)
norm_2 = lambda x : np.linalg.norm(x, 2)
norm_F = lambda x : np.linalg.norm(x, 'fro')

# penalty function \phi
phi = lambda x : norm_1(x)


# float comparison threshold
THRESH = 1e-7

def is_zero(x, thresh = THRESH):
	return abs(x) < thresh

# sign function
sign = lambda x : int(x / abs(x)) if not is_zero(x) else 0



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
		if (temp := func(arr[i])) < min_val:
			min_val = temp
			min_idx = i
	
	return min_idx,min_val
	




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
	return argmin_f(W, func)[1]

