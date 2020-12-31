
## Run project

> python nn_train.py data_training.csv

### Proceed training

> python nn_train.py data_training.csv -w weights.csv

### Set architecture for neural network (neurons for hidden layers: 1 and 2)

> python nn_train.py data_training.csv -a 20 10

### Proceed training with new architecture

> python nn_train.py data_training.csv -w weights -a 20 10

### Prediction

> python nn_predict.py data_test.csv weights.csv

### Prediction with new architecture

> python nn_predict.py data_test.csv weights.csv -a arch.csv

### Prediction with confusion matrix

> python nn_predict.py data_test.csv weights.csv -c



## Bonuses

1. MinMaxScaler - selfmade

2. accuracy while training

3. N_epochs input:

    > python nn_train.py data_training.csv -e 1000 

4. Confusion matrix:

5. Plot for train and validation data: loss and accuracy

6. Precision and recall
