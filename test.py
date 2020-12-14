import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm

def derivative_hidden_test(target, output, weights_2, X, n_neurons):
    gradients_w = []
    gradients_b = []
    for i in range(n_neurons):
        summa_w = 0.0
        summa_b = 0.0
        for j in range(len(X)):
            summa_w += (-2.) * (target[j] - output.iloc[j, 0]) * weights_2.iloc[i, 0] * (np.exp(X.iloc[j, 0]) / (1 + np.exp(X.iloc[j, 0]))) * X.iloc[j, 0]
            summa_b += (-2.) * (target[j] - output.iloc[j, 0]) * weights_2.iloc[i, 0] * (np.exp(X.iloc[j, 0]) / (1 + np.exp(X.iloc[j, 0])))
        gradients_w.append(summa_w)
        gradients_b.append(summa_b)
    return np.array(gradients_w), np.array(gradients_b)


# def derivative_hidden(target, output, weights_2, X, n_neurons):
# 	gradients_w = []
# 	gradients_b = []
# 	for i in range(n_neurons):
# 		dSSR_dPred = np.sum((-2.) * (target - output))
# 		dPred_dy = weights_2.iloc[i, 0]
# 		dy_dx = np.array(list(map(lambda x: np.exp(x) / (1 + np.exp(x)), X.T))).reshape(X.shape[1], X.shape[0])
# 		dx_dw = X
# 		dSSR_dw = np.dot(np.dot(np.dot(dSSR_dPred, dPred_dy), dy_dx), dx_dw)
#
# 		dSSR_db = np.dot(np.dot(np.dot(dSSR_dPred, dPred_dy), dy_dx), np.ones(X.shape))
#
# 		gradients_w.append(dSSR_dw)
#
# 		gradients_b.append(dSSR_db)
# 	return np.array(gradients_w), np.array(gradients_b)


def derivative_hidden(target, output, weights_2, X, n_neurons):
	gradients_w = []
	gradients_b = []
	for i in range(n_neurons):
		dSSR_dPred = np.sum((-2.) * (target - output))
		dPred_dy = weights_2.iloc[i, 0]
		dy_dx = np.array(list(map(lambda x: np.exp(x) / (1 + np.exp(x)), X.T))).reshape(X.shape[1], X.shape[0])
		dx_dw = X
		dSSR_dw = (dSSR_dPred * dPred_dy) * np.dot(dy_dx, dx_dw)[0][0]

		dSSR_db = np.dot(np.dot(np.dot(dSSR_dPred, dPred_dy), dy_dx), np.ones(X.shape))

		gradients_w.append(dSSR_dw)

		gradients_b.append(dSSR_db)
	return pd.DataFrame(np.array(gradients_w)), pd.DataFrame(np.array(gradients_b))


def derivative_weights(target, output, hidden1):
    derivative = []
    for i in range(1, hidden1.shape[1] + 1):
        derivative.append(np.dot((-2.) * (target - output).T, hidden1['n' + str(i)]))
    return np.array(derivative).reshape(-1, output.shape[1])

def derivative_of_b3(target, output):
    return np.sum((-2) * (target - output))

def ssr(target, predicted):
    return np.sum((target - predicted) ** 2)

def softmax(x):
    return np.log(1 + np.exp(x))

def crossenthropy(x):
    return 1. / (1. + np.exp(-x))

def neuron_layer(X, weights, b, activation=None):
    Z = np.dot(X, weights) + b
    if activation is not None:
        Z = activation(Z)
        pass
    Z = pd.DataFrame(Z,
                     columns=['n' + str(i) for i in range(1, weights.shape[1] + 1)],
                     index = [str(i) + ' object' for i in range(1, X.shape[0] + 1)])
    return Z

def init_weights(X, n_neurons, random_state=42):
	n_inputs = int(X.shape[1])
	stddev = 2 / np.sqrt(n_inputs + n_neurons)
	b = np.zeros(n_neurons)
	weights = pd.DataFrame(truncnorm.rvs(-1, 1, size=(n_inputs, n_neurons), scale=stddev, random_state=random_state),
						   columns=[str(i) for i in range(1, n_neurons + 1)],
						   index=['w' + str(x) for x in range(1, n_inputs + 1)])

	return pd.DataFrame(weights), b


epochs = 10


def train(X, target, epochs=10, learning_rate=0.1):
	weights_1, biases_1 = init_weights(X, 2)

	hidden1 = neuron_layer(X, weights_1, biases_1, activation=None)

	weights_2, biases_2 = init_weights(hidden1, 1, random_state=43)
	output = neuron_layer(hidden1, weights_2, biases_2, activation=None)
	for epoch in range(epochs):
		error = ssr(target, output)
		print(epoch+1, ": Error:", np.array(error)[0])

		## b3
		gradient_b3 = derivative_of_b3(target, output)
		step_size_b3 = gradient_b3 * learning_rate
		biases_2 = np.array(biases_2 - step_size_b3)

		##-- w3, w4
		gradient_weights = derivative_weights(target, output, hidden1)
		step_size_weights = gradient_weights * learning_rate
		weights_2 = weights_2 - step_size_weights

		##-- w1, w2, b1, b2
		gradient_weights_1_2, gradient_biases_1_2 = derivative_hidden(target, output, weights_2, X, 2)
		step_size_w_1_2 = gradient_weights_1_2 * learning_rate
		step_size_b_1_2 = gradient_biases_1_2 * learning_rate
		weights_1 = weights_1 - step_size_w_1_2.T
		biases_1 = biases_1 - step_size_b_1_2.T

		hidden1 = neuron_layer(X, weights_1, biases_1, activation=crossenthropy)
		output = neuron_layer(hidden1, weights_2, biases_2, activation=None)

		pass

	print(output)


X = pd.DataFrame(np.linspace(0, 1, num=3).reshape(-1, 1), columns=['Dosage'])
target = np.array([0, 1, 0]).reshape(-1, 1)

train(X, target, epochs=10, learning_rate=0.1)