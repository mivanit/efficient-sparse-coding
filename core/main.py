'''
python implementation of the algorithms described in the paper
"Efficient sparse coding algorithms"
by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

for Math 651 at the University of Michigan, Winter 2020

by Michael Ivanitskiy and Vignesh Jagathese

USAGE:
	two modes:
		[default]  :  manual / specify all args

		-f / --file  :  specify settings file
							(can still specify args in command line, will overwrite)

	valid settings file should contain values for:
		n  		:  num basis vectors
		gamma 	:  constant
		c_const :  constrainst constant
		delta   :  iteration termination threshold
		file_X  :  path to array file
'''

import sys
import numpy as np

from SparseCoder import *
# from util import *

#test comment yeet


DICT_SAVE_SPLITCHAR = {
	'csv' : ',',
	'tsv' : '\t',
}


 ######  ########  ######
##    ## ##       ##    ##
##       ##       ##
##       ######   ##   ####
##       ##       ##    ##
##    ## ##       ##    ##
 ######  ##        ######


def arg_val(key_set, argv = sys.argv, val = True):
	for k in key_set:
		if k in argv:
			i = argv.index(k)
			return argv[i+1]

	# if not found
	return None


def arg_val_assign(key_set, var, argv = sys.argv):
	temp = arg_val(key_set, argv)
	if temp is not None:
		var = temp



def load_setttings(file_set):
	'''
	valid settings file should contain values for:
		n  		:  num basis vectors
		gamma 	:  constant
		c_const :  constrainst constant
		delta   :  iteration termination threshold
		file_X  :  path to array file
	'''
	output = dict()
	with open(file_set, 'r') as f:
		for line in f:
			a,b = line.split(':')
			output[a.strip()] = b.strip()

	return output


def argParser(argv = sys.argv):

	# see if settings file given
	settings_file = arg_val(['-f', '--file'], argv)
	
	# read settings from file
	if settings_file is not None:
		settings = load_setttings(settings_file)
	else:
		settings = dict()
	
	# read (overwriting if present) settings from command line args

	# solver vars
	arg_val_assign(['-n'], settings.get('n'), argv)
	arg_val_assign(['-g', '--gamma'],  settings.get('gamma'), argv)
	arg_val_assign(['-c', '--c_const'],  settings.get('c_const'), argv)
	arg_val_assign(['-d', '--delta'],  settings.get('delta'), argv)
	
	# file paths
	arg_val_assign(['--file_X'],  settings.get('file_X'), argv)
	arg_val_assign(['--file_B'],  settings.get('file_B'), argv)
	arg_val_assign(['--file_S'],  settings.get('file_S'), argv)


	# test for required settings present
	if any([
		settings.get(s, None) is None
		for s in 'n,gamma,c_const,delta,file_X'.split(',')
	]):
		print(__doc__)
		print('='*50)
		raise Exception('Missing a required argument! exiting program')

	# change types
	for s in 'gamma,c_const,delta'.split(','):
		settings[s] = float(settings[s])

	settings['n'] = int(settings['n'])

	# output files
	for s in ['file_B', 'file_S', 'out_fmt']:
		if settings.get(s, None) is None:
			settings[s] = None
			
	return settings

	

########     ###    ########    ###
##     ##   ## ##      ##      ## ##
##     ##  ##   ##     ##     ##   ##
##     ## ##     ##    ##    ##     ##
##     ## #########    ##    #########
##     ## ##     ##    ##    ##     ##
########  ##     ##    ##    ##     ##


def load_X(file_in):
	# this doesnt work if you want to move up a directory (or if there are periods in file/folder names)
	# file_in_name, file_in_type = file_in.split('.')[-2:]
	idx_temp = file_in.rfind('.')
	file_in_name = file_in[:idx_temp]
	file_in_type = file_in[idx_temp+1:]


	if file_in_type == 'npy':
		# load numpy binary
		arr_X = np.load(file_in)

	elif file_in_type in DICT_SAVE_SPLITCHAR:
		# load text csv
		splitchar = DICT_SAVE_SPLITCHAR[file_in_type]
		
		arr_X = [ None ]
		with open(file_in, 'r') as f:
			linenum = 0
			for line in f:
				arr_X[linenum] = [ float(s) for s in line.split( splitchar ) ]
				arr_X.append(None)
		arr_X = np.array(arr_X)
	
	else:
		raise Exception('invalid format:\t%s' % file_in_type)

	return arr_X


def save_results(results, settings):
	res_B, res_S = results

	idx_temp = settings['file_X'].rfind('.')
	file_in_name = settings['file_X'][:idx_temp]
	file_in_type = settings['file_X'][idx_temp+1:]

	if settings['out_fmt'] is None:
		settings['out_fmt'] = file_in_type

	if settings['file_B'] is None:
		settings['file_B'] = file_in_name + '_B' + '.' + settings['out_fmt']

	if settings['file_S'] is None:
		settings['file_S'] = file_in_name + '_S' + '.' + settings['out_fmt']

	save_arr(res_B, settings['file_B'], settings['out_fmt'])
	save_arr(res_S, settings['file_S'], settings['out_fmt'])


def save_arr(arr, name, fmt):
	if fmt in DICT_SAVE_SPLITCHAR:
		np.savetxt(name, arr, delimiter = DICT_SAVE_SPLITCHAR[fmt])
	
	elif fmt == 'npy':
		np.save(name, arr)

	else:
		raise Exception('invalid format:\t%s' % fmt)

	





##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##

def main(argv = sys.argv):

	print('> reading settings')
	cfg = argParser(argv)

	print('> reading input array')
	arr_X = load_X(cfg['file_X'])

	cfg['k'],cfg['m'] = arr_X.shape

	print('SETTINGS:')
	for key,val in cfg.items():
		print('\t%s\t:\t%s\t:\t%s' % (str(key), str(val), str(type(val))))
	print('-'*60)

	print('> setting up solver')
	coder = SparseCoder(cfg['n'], cfg['k'], arr_X, cfg['c_const'], cfg['gamma'])

	print('> solving')
	res = coder.train(cfg['delta'], True)

	print('> done! iterations: \t%d' % res['iters'])

	print('> saving results')
	save_results( (res['B'], res['S']), cfg )


	
main()



	


