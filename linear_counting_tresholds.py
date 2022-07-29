import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
import numpy as np
import pandas as pd
import sys

if len(sys.argv) != 2:
	print("Usage: {sys.argv[0]} <bias_data_file>")
	sys.exit(1)

frame = pd.read_csv(sys.argv[1], sep = ',', skipinitialspace = True)

min_prec = 6
max_prec = 18
n_prec = max_prec - min_prec + 1
err_data = list(frame['std_err'])
card_data = list(frame['card'])

err_lists = np.array_split(err_data, n_prec)
card_lists = np.array_split(card_data, n_prec)

def curve(x, a, b, c, d):
	return a * x**3 + b * x**2 + c * x**1 + d

with open('linear_counting_treshonds', 'w') as thresholds_file:
	for prec in range(min_prec, max_prec + 1):
		m = 2 ** prec
		err = 1.04 / np.sqrt(m)
		idx = prec - min_prec
		coefs, errors = curve_fit(curve, card_lists[idx], err_lists[idx])

		error_curve = lambda x : curve(x, *coefs)
		error_limit = err
		solution = root_scalar(lambda x : error_curve(x) - error_limit, bracket=[0, 3 * m])

		thresholds_file.write(f'\t/* precision {prec} */\n')
		thresholds_file.write(f'\t{int(solution.root)},\n')
