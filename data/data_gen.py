import sys
import numpy as np





def uniform(params):
	k,m,b = params
	k = int(k)
	m = int(m)
	b = float(b)

	return b * np.random.rand(k,m)









def main(argv = sys.argv):
	mode, fname, *params = argv[1:]

	arr = None
	if mode in ['u', '-u', '--uniform']:
		arr = uniform(params)

	np.save(fname, arr)

main()
	
