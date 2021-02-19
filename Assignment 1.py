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
        for i in range(self.num_classes):
            prior = len(Xtrain[ytrain == i]) / len(Xtrain)
            for j in range(len(self.feature_types)):
                dist = self.find_condi_distribution(Xtrain[ytrain == i][:, j])


        return self.class_distribution

# classify one example
Xsample, ysample = X[0], y[0]
py0 = probability(Xsample, priory0, distX1y0, distX2y0)
py1 = probability(Xsample, priory1, distX1y1, distX2y1)
print('P(y=0 | %s) = %.3f' % (Xsample, py0*100))
print('P(y=1 | %s) = %.3f' % (Xsample, py1*100))
print('Truth: y=%d' % ysample)

    def predict(self, Xtest):
        return 0

    def find_condi_distribution(self, c):
        class_mean = np.mean(c)
        sigma = np.std(c)
        print(class_mean, sigma)
        # fit distribution
        dist = norm(class_mean, sigma)
        return dist
    
# calculate the independent conditional probability
def probability(X, prior, dist1, dist2):
    return prior * dist1.pdf(X[0]) * dist2.pdf(X[1])


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