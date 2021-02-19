from sklearn.datasets import load_iris
import numpy as np

class NBC():
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes

    def fit(self, Xtrain, ytrain):

        return

    def predict(self, Xtest):
        return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load iris dataset
    iris = load_iris()
    X, y = iris['data'], iris['target']

    # shuffle the dataset and put 20% aside for testing
    N, D = X.shape
    Ntrain = int(0.8 * N)
    shuffler = np.random.permutation(N)
    Xtrain = X[shuffler[:Ntrain]]
    ytrain = y[shuffler[:Ntrain]]
    Xtest = X[shuffler[Ntrain:]]
    ytest = y[shuffler[Ntrain:]]

    # initialize the Naive Bayes Classifier
    nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)

    # fit the classifier to training data
    nbc.fit(Xtrain, ytrain)

    # predict the classes
    yhat = nbc.predict(Xtest)

    # evaluate the result
    test_accuracy = np.mean(yhat == ytest)