import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm


def neuron_layer(X, n_neurons, activation=None):
    n_inputs = int(X.shape[1])
    stddev = 2 / np.sqrt(n_inputs + n_neurons)
    weights = pd.DataFrame(truncnorm.rvs(-1, 1, size=(n_inputs, n_neurons), scale=stddev, random_state=42),
                           columns = ['1', '2'],
                           index=['w' + str(x) for x in range(1, n_inputs + 1)])

    b = np.ones(n_neurons)
    Z = np.dot(X, weights) + b
    if activation is not None:
        return activation(Z)
    else:
        return Z


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv"
df = pd.read_csv(url, header=None)
df.columns = ['area',
              'perimeter',
              'compactness',
              'length of kernel',
              'width of kernel',
              'asymmetry coefficient',
              'length of kernel groove',
             'target']

neuron_layer(df, 2)

from random import seed
from random import random


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
    print(layer)