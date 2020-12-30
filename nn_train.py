import pandas as pd
import argparse
from nn.preprocess_data import preprocess
from nn.train_test_split import train_test_split
from nn.nn_model import NeuralNetwork
from nn.nn_model import ft_accuracy_score
from nn.confusion_matrix import plotCf
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help="input dataset")
	parser.add_argument('-e', '--epochs', nargs=1, type=int, help="Option epochs")
	parser.add_argument('-p', '--predict', action="store_true", help="Predict")
	parser.add_argument('-c', '--confusion', action="store_true", help="Confusion matrix")
	args = parser.parse_args()

	df = pd.read_csv(args.dataset, names=['1','target','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32'])
	X, y = preprocess(df)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	X_valid, y_valid = X_train[:50], y_train[:50]
	X_train, y_train = X_train[50:], y_train[50:]

	if args.epochs == None:
		n_epochs = 2000
	else:
		n_epochs = args.epochs[0]


	NN = NeuralNetwork(20, 10, 2, 0.1)
	NN.fit(X_train, y_train, n_epochs=n_epochs, valid=(X_valid, y_valid))

	if args.predict:
		##-- train data
		y_pred_train = NN.predict(X_train, y_train)

		##-- test data
		y_pred = NN.predict(X_test, y_test)

		print("{}: {}".format("\nAccuracy (train)", ft_accuracy_score(y_train, y_pred_train)))
		print("{}: {}".format("Accuracy (test)", ft_accuracy_score(y_test, y_pred)))

		##-- confusion matrix (train)
		cf_train = confusion_matrix(y_train, y_pred_train)
		precision_train, recall_train = cf_train[1][1] / (cf_train[1][1] + cf_train[0][1]), \
										cf_train[1][1] / (cf_train[1][1] + cf_train[0][0])

		##-- confusion matrix (test)
		cf_test = confusion_matrix(y_test, y_pred)
		precision_test, recall_test = cf_test[1][1] / (cf_test[1][1] + cf_test[0][1]), \
										cf_test[1][1] / (cf_test[1][1] + cf_test[0][0])
		print("(Train) Precision: {}, Recall: {}".format(precision_train, recall_train))
		print("(Test) Precision: {}, Recall: {}".format(precision_test, recall_test))

		if args.confusion:
			plotCf(y_test, y_pred, "Confusion matrix(test)")