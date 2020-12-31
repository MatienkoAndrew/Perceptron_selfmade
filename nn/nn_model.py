# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    nn_model.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: student <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/12/30 08:11:05 by student           #+#    #+#              #
#    Updated: 2020/12/30 08:11:08 by student          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd

from scipy.stats import truncnorm
import matplotlib.pyplot as plt

def ft_accuracy_score(y, y_pred):
    y = y.values
    y_pred = y_pred
    summa = 0.0
    m = len(y)
    for i in range(m):
        if y[i] == y_pred[i]:
            summa += 1
    return summa / m

class NeuralNetwork:
    def __init__(self, inputs_neurons, hidden_neurons1, hidden_neurons2, out_neurons, learning_rate=0.1, weights=None):
        self.inputs_neurons = inputs_neurons
        self.hidden_neurons1 = hidden_neurons1
        self.hidden_neurons2 = hidden_neurons2
        self.out_neurons = out_neurons
        self.weights = weights

        self.param = {}
        self.cach = {}
        self.loss = []
        self.loss_valid = []
        self.lr = learning_rate

        if self.weights is not None:
            try:
                self.param['W1'] = self.weights[0].reshape(hidden_neurons1, self.inputs_neurons)
                self.param['b1'] = pd.DataFrame(self.weights[1]).dropna().values.reshape(-1, 1)
                self.param['W2'] = pd.DataFrame(self.weights[2]).dropna().values.reshape(hidden_neurons2, hidden_neurons1)
                self.param['b2'] = pd.DataFrame(self.weights[3]).dropna().values.reshape(-1, 1)
                self.param['W3'] = pd.DataFrame(self.weights[4]).dropna().values.reshape(out_neurons, hidden_neurons2)
                self.param['b3'] = pd.DataFrame(self.weights[5]).dropna().values.reshape(-1, 1)
            except:
                print("Use another architecture")
                exit(1)


    def init_bias(self, n_neurons):
        b = np.zeros(n_neurons)
        return b.reshape(-1, 1)

    def init_weights(self, n_inputs, n_neurons):
        random_state = np.random.randint(1000)
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        weights = truncnorm.rvs(-1, 1, size=(n_neurons, n_inputs), scale=stddev, random_state=random_state)
        return weights

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def dsigmoid(self, Z):
        return Z * (1 - Z)

    def Relu(self, Z):
        return np.maximum(0, Z)

    def dRelu2(self, dZ, Z):
        dZ[Z <= 0] = 0
        return dZ

    def dRelu(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def softmax(self, Z):
        shiftx = Z - np.max(Z, axis=1).reshape(-1, 1)
        e = np.exp(shiftx)
        s = np.sum(e, axis=1).reshape(-1, 1)
        return e / s

    def dsoftmax(self, Z):
        p = self.cach['f3']
        dA = (Z * p).sum(axis=1).reshape(-1, 1)
        return p * (Z - dA)

    def init(self, m):
        self.param['W1'] = self.init_weights(m, self.hidden_neurons1)
        self.param['b1'] = self.init_bias(self.hidden_neurons1)
        self.param['W2'] = self.init_weights(self.hidden_neurons1, self.hidden_neurons2)
        self.param['b2'] = self.init_bias(self.hidden_neurons2)
        self.param['W3'] = self.init_weights(self.hidden_neurons2, self.out_neurons)
        self.param['b3'] = self.init_bias(self.out_neurons)
        pass

    def crossenthropy(self, y, y_pred):
        m = y.shape[1]
        return (-1.0 / m) * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def feedforward(self, X, weights, bias, activation=None):
        V = np.dot(weights, X) + bias
        if activation is not None:
            F = activation(V)
        return V, F

    def forward(self, X, y, activation=None):
        v1, f1 = self.feedforward(X, self.param['W1'], self.param['b1'], self.Relu)
        v2, f2 = self.feedforward(f1, self.param['W2'], self.param['b2'], self.Relu)
        v3, f3 = self.feedforward(f2, self.param['W3'], self.param['b3'], self.softmax)
        y_pred = f3

        self.cach['v1'] = v1
        self.cach['f1'] = f1
        self.cach['v2'] = v2
        self.cach['f2'] = f2
        self.cach['v3'] = v3
        self.cach['f3'] = f3

        loss = self.crossenthropy(y, y_pred)

        return y_pred, loss

    def backpropagation(self, X, y, y_pred):
        derror = -(np.divide(y, y_pred) - np.divide(1 - y, 1 - y_pred))

        # local_grad1 = derror * self.dsigmoid(y_pred)
        local_grad1 = self.dsoftmax(derror)
        grad_w3 = 1.0 / self.cach['f2'].shape[1] * np.dot(local_grad1, self.cach['f2'].T)
        grad_b3 = 1.0 / self.cach['f2'].shape[1] * np.dot(local_grad1, np.ones([local_grad1.shape[1], 1]))
        hidden_error1 = np.dot(self.param["W3"].T, local_grad1)

        local_grad2 = hidden_error1 * self.dRelu(self.cach['f2'])
        grad_w2 = 1. / self.cach['f1'].shape[1] * np.dot(local_grad2, self.cach['f1'].T)
        grad_b2 = 1. / self.cach['f1'].shape[1] * np.dot(local_grad2, np.ones([local_grad2.shape[1], 1]))
        hidden_error2 = np.dot(self.param['W2'].T, local_grad2)

        local_grad3 = hidden_error2 * self.dRelu(self.cach['f1'])
        grad_w1 = 1. / X.shape[1] * np.dot(local_grad3, X.T)
        grad_b1 = 1. / X.shape[1] * np.dot(local_grad3, np.ones([local_grad3.shape[1], 1]))

        self.param["W1"] -= self.lr * grad_w1
        self.param["b1"] -= self.lr * grad_b1
        self.param["W2"] -= self.lr * grad_w2
        self.param["b2"] -= self.lr * grad_b2
        self.param["W3"] -= self.lr * grad_w3
        self.param["b3"] -= self.lr * grad_b3

        pass

    def preprocess_X_y(self, X, y):
        X = X.T.values

        y = pd.DataFrame(y).copy()
        y[0] = y['target'].apply(lambda x: 0 if x == 1 else 1)
        y[1] = y['target'].apply(lambda x: 1 if x == 1 else 0)
        y.drop(['target'], axis=1, inplace=True)
        y = y.T.values

        return X, y

    def fit(self, X, y, n_epochs=3000, valid=None):
        m = X.shape[1]
        if valid != None:
            X_valid, y_valid = valid[0], valid[1]
            X_valid_for_accuracy = X_valid
            y_valid_for_accuracy = y_valid

        if self.weights is None:
            self.init(m)

        X_for_accuracy = X
        y_for_accuracy = y

        X, y = self.preprocess_X_y(X, y)

        i_list = []
        accuracy_list = []
        if valid != None:
            accuracy_valid_list = []

        for i in range(1, n_epochs + 1):
            if valid != None:
                y_proba_valid, loss_valid = self.predict_proba(X_valid, y_valid)

            y_proba, loss = self.forward(X, y)
            self.backpropagation(X, y, y_proba)

            ##-- loss
            if valid != None:
                loss_valid = np.sum(np.sum(loss_valid, axis=1))
            loss = np.sum(np.sum(loss, axis=1))

            ##-- predictions
            if valid != None:
                y_pred_valid = self.predict(X_valid_for_accuracy, y_valid_for_accuracy)
            y_pred = self.predict(X_for_accuracy, y_for_accuracy)

            ##-- accuracy
            if valid != None:
                accuracy_valid = ft_accuracy_score(y_valid_for_accuracy, y_pred_valid)
            accuracy = ft_accuracy_score(y_for_accuracy, y_pred)

            ##-- print
            if valid == None:
                print("epoch {0:>5}/{1} - loss: {2:<6.4} - accuracy: {3:<6.4}".format(i, n_epochs, loss, accuracy))
            else:
                print(
                    "epoch {0:>5}/{1} - loss: {2:<6.4} - accuracy: {3:<6.4} - val_loss: {4:<6.4} - val_accuracy: {5:.4}".format(
                        i, n_epochs, loss,
                        accuracy, loss_valid,
                        accuracy_valid))

            ##-- append
            if valid != None:
                self.loss_valid.append(loss_valid)
                accuracy_valid_list.append(accuracy_valid)
            self.loss.append(loss)
            i_list.append(i)
            accuracy_list.append(accuracy)

        plt.figure(figsize=(16, 6))
        plt.plot(i_list, self.loss, label='loss')
        if valid != None:
            plt.plot(i_list, self.loss_valid, label='val_loss')
            plt.plot(i_list, accuracy_valid_list, label='val_accuracy')
        plt.plot(i_list, accuracy_list, label='accuracy')

        plt.title("Learning rate =" + str(self.lr))
        plt.legend()
        plt.grid(True)
        plt.show()

        ##--save weights
        w1 = self.param["W1"].reshape(1, -1)
        b1 = self.param["b1"].reshape(1, -1)
        w2 = self.param["W2"].reshape(1, -1)
        b2 = self.param["b2"].reshape(1, -1)
        w3 = self.param["W3"].reshape(1, -1)
        b3 = self.param["b3"].reshape(1, -1)

        self.weights = [w1.tolist()[0],
                        b1.tolist()[0],
                        w2.tolist()[0],
                        b2.tolist()[0],
                        w3.tolist()[0],
                        b3.tolist()[0]]

        return

    def predict_proba(self, X, y):
        X, y = self.preprocess_X_y(X, y)
        y_pred, loss = self.forward(X, y)
        return y_pred, loss

    def predict(self, X, y):
        y_pred, _ = self.predict_proba(X, y)
        predict = np.array([x for x in y_pred.argmax(axis=0)])
        return predict

