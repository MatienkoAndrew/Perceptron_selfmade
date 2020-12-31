import pandas as pd
import numpy as np
import argparse
from nn.preprocess_data import preprocess
from nn.nn_model import NeuralNetwork
from nn.nn_model import ft_accuracy_score
from nn.confusion_matrix import plotCf
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help="input dataset")
	parser.add_argument('weights', type=str, help="input weights")
	parser.add_argument("-a", "--architecture", type=str, help="input architecture")
	parser.add_argument('-c', '--confusion', action="store_true", help="Confusion matrix")
	args = parser.parse_args()

	df = pd.read_csv(args.dataset, names=['1','target','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32'])
	X_test, y_test = preprocess(df)

	weights = pd.read_csv(args.weights)
	weights = weights.values

	if args.architecture != None:
		if args.architecture != 'arch.csv':
			print("Need file 'arch.csv'")
			exit(1)
		architecture = pd.read_csv('arch.csv')
		architecture = np.squeeze(architecture.values)
		hidden_neurons1 = architecture[0]
		hidden_neurons2 = architecture[1]
	else:
		hidden_neurons1 = 20
		hidden_neurons2 = 10
	NN = NeuralNetwork(X_test.shape[1], hidden_neurons1, hidden_neurons2, 2, 0.1, weights=weights)

	##-- test data
	y_pred = NN.predict(X_test, y_test)

	print("{}: {}".format("Accuracy", ft_accuracy_score(y_test, y_pred)))

	##-- confusion matrix (test)
	cf_test = confusion_matrix(y_test, y_pred)
	try:
		precision_test = int(cf_test[1][1]) / int(cf_test[1][1] + cf_test[0][1])
		recall_test = int(cf_test[1][1]) / int(cf_test[1][1] + cf_test[0][0])
		print("(Test) Precision: {}, Recall: {}".format(precision_test, recall_test))
	except ZeroDivisionError:
		print("Не можем показать Precision/Recall: Деление на ноль")
	except:
		print("ERROR")

	if args.confusion:
		plotCf(y_test, y_pred, "Confusion matrix(test)")