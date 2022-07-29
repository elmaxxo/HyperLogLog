import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
est_data = list(frame['avg_est'])
card_data = list(frame['card'])

bias_data = list(np.subtract(est_data, card_data))

est_lists = np.array_split(est_data, n_prec)
bias_lists = np.array_split(bias_data, n_prec)

def curve(x, a, b, c, d, e, f):
	return a * (x**5) + b * (x**4) + c * (x**3) + d * (x**2) + e * (x**1) + f

with open('bias_corrections_curves', 'w') as curves_file:
	for prec in range(min_prec, max_prec + 1):
		idx = prec - min_prec
		coefs, errors = curve_fit(curve, est_lists[idx], bias_lists[idx])

		curves_file.write(f'/* precision {prec} */\n')
		curves_file.write('{\n')

		for coef in coefs:
			curves_file.write(f"\t{coef},\n")

		curves_file.write('},\n')
