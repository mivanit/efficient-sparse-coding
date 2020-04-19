import sys
import numpy as np





def uniform(params):
	k,m,b = params
	k = int(k)
	m = int(m)
	b = float(b)

	return b * np.random.rand(k,m)

def uniform_z(params):
	'''
	k,m is dims
	b is a const
	z is the number of zero elements in each vector
	'''
	
	k,m,b,z = params
	k = int(k)
	m = int(m)
	b = float(b)
	z = int(z)

	output = np.concatenate([
		b * np.random.rand(k-z,m),
		np.zeros((z,m))
	], 0)
	print(output)
	return output







def main(argv = sys.argv):
	mode, fname, *params = argv[1:]

	arr = None
	if mode in ['u', '-u', '--uniform']:
		arr = uniform(params)
	elif mode in ['uz', '-z']:
		arr = uniform_z(params)
	else:
		raise Exception('invalid mode')

	np.save(fname, arr)

main()
	
