
from sklearn.datasets import load_iris
import numpy as np
from scipy.stats import norm

class NBC():
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes
        self.classes = []
        self.priors = {}
        self.dists = {}

    def fit(self, X, y):
        self.classes = np.unique(y) # find unique classes
        for i in self.classes:
            prior = len(X[y == i]) / len(X) # find prior probabilities
            self.priors[i] = prior # add prior probabilities to a list
            dist_list = []
            for j in range(len(self.feature_types)):
                dist = self.find_class_distribution(X[y == i][:, j]) # calculate the distribution for different features
                dist_list.append(dist)  # add distribution of features to a list
            self.dists[i] = dist_list


    def predict(self, X):
        prediction = [] # [0, 0, 1, 2, 3...]
        for data_point in X:
            class_probs = {}
            for i in self.classes:
                prob = self.priors[i] # get the prior probabilities for classes
                for j in range(len(self.feature_types)):
                    prob = prob * self.dists[i][j].pdf(data_point[j]) # get the probabilities for different features
                class_probs[i] = prob  # class_probs = {0: 0.5, 1: 0.2, 2: 0.2, 3: 0.1}
            highest = max(class_probs, key=class_probs.get) # find the highest probability for class prediction
            prediction.append(highest)
        return prediction


    def find_class_distribution(self, c):
        # calculate the class distribution
        c_mean = np.mean(c)
        sigma = np.std(c)
        # fit class distribution
        dist = norm(c_mean, sigma)
        return dist


if __name__ == "__main__":
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
    yhat_train = nbc.predict(Xtrain)

    # evaluate the result
    train_accuracy = np.mean(yhat_train == ytrain)
    test_accuracy = np.mean(yhat == ytest)

    print("train_accuracy: " +str(train_accuracy))
    print("yhat: "+str(yhat))
    print("test_accuracy: " +str(test_accuracy))