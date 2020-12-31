# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    nn_train.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: student <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/12/30 08:08:02 by student           #+#    #+#              #
#    Updated: 2020/12/30 08:09:20 by student          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import argparse
from nn.preprocess_data import preprocess
from nn.nn_model import NeuralNetwork

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help="input dataset")
	parser.add_argument('-e', '--epochs', nargs=1, type=int, help="Option epochs")
	parser.add_argument("-w", "--weights", nargs=1, type=str, help="input weights")
	parser.add_argument("-a", "--architecture", nargs=2, type=int, help="input architecture")
	args = parser.parse_args()

	df = pd.read_csv(args.dataset, names=['1','target','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32'])
	X, y = preprocess(df)

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	X_valid, y_valid = X[:50], y[:50]
	X_train, y_train = X[50:], y[50:]

	if args.epochs == None:
		n_epochs = 2000
	else:
		n_epochs = args.epochs[0]

	if args.weights == None:
		weights = None
	else:
		if args.weights[0] != 'weights.csv':
			print("Need file 'weights.csv'")
			exit(1)
		weights = pd.read_csv(args.weights[0])
		weights = weights.values

	if args.architecture != None:
		hidden_neurons1 = args.architecture[0]
		hidden_neurons2 = args.architecture[1]
	else:
		hidden_neurons1 = 20
		hidden_neurons2 = 10

	NN = NeuralNetwork(X_train.shape[1], hidden_neurons1, hidden_neurons2, 2, learning_rate=0.1, weights=weights)
	NN.fit(X_train, y_train, n_epochs=n_epochs, valid=(X_valid, y_valid))

	pd.DataFrame(NN.weights).to_csv('weights.csv', index=False)

	if args.architecture != None:
		pd.DataFrame(args.architecture).to_csv('arch.csv', index=False)
	pass
