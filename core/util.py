import numpy as np

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