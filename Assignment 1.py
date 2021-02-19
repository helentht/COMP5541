from sklearn.datasets import load_iris

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load iris dataset
    iris = load iris()
    X, y = iris[‘data’], iris[‘target’]

    # shuffle the dataset and put 20% aside for testing
    N, D = X.shape
    Ntrain = int(0.8 * N)
    shuffler = np.random.permutation(N) Xtrain = X[shuffler[:Ntrain]]
    ytrain = y[shuffler[:Ntrain]]
    Xtest = X[shuffler[Ntrain:]]
    ytest = y[shuffler[Ntrain:]]
    
    nbc = NBC(feature types=[‘r’, ‘r’, ‘r’, ‘r’], num classes=3)
    nbc.fit(Xtrain, ytrain)
    yhat = nbc.predict(Xtest)
    test accuracy = np.mean(yhat == ytest)