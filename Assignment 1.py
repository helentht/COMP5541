from sklearn.datasets import load_iris
import numpy as np
from scipy.stats import norm

class NBC():
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes
        self.classes = []
        self.length = 0
        self.class_count = {}
        self.class_distribution = {}


    def fit(self, Xtrain, ytrain):
        self.length = len(ytrain)

        unique, counts = np.unique(ytrain, return_counts=True)
        self.classes = unique
        self.class_count = dict(zip(unique, counts))
        for key, value in self.class_count.items():
            self.class_distribution[key] = value/self.length

        print(Xtrain)
        for c in self.classes:
            return 0

        return self.class_distribution

    def predict(self, Xtest):
        return 0

    def find_condi_distribution(self, c):
        class_mean = np.mean(c)
        sigma = np.std(c)
        print(class_mean, sigma)
        # fit distribution
        dist = norm(class_mean, sigma)
        return dist

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
    a = nbc.fit(Xtrain, ytrain)
    print(a)

    # predict the classes
    yhat = nbc.predict(Xtest)

    # evaluate the result
    test_accuracy = np.mean(yhat == ytest)