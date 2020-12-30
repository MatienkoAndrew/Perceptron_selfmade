# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    min_max_scaler.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: student <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/12/30 08:09:37 by student           #+#    #+#              #
#    Updated: 2020/12/30 08:09:42 by student          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd

class MinMaxScaler:
	def __init__(self):
		self.min_val = None
		self.max_val = None
		pass

	def max_vals(self, df_col):
		max_v = -100000000.0
		for x in df_col:
			if x > max_v:
				max_v = x
		return max_v

	def min_vals(self, df_col):
		min_v = 100000000.0
		for x in df_col:
			if x < min_v:
				min_v = x
		return min_v

	def fit(self, X):
		self.min_val, self.max_val = [], []
		for col in X.columns:
			self.min_val.append(self.min_vals(X[col]))
			self.max_val.append(self.max_vals(X[col]))
		pass

	def fit_transform(self, X):
		self.fit(X)
		X_temp = X.copy()

		for i, col in enumerate(X_temp.columns):
			vals_scaled = []
			for x in X_temp[col]:
				vals_scaled.append((x - self.min_val[i]) / (self.max_val[i] - self.min_val[i]))
			X_temp[col] = vals_scaled
		return pd.DataFrame(X_temp)
