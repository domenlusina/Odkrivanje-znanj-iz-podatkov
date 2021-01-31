import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets


class Kernel:
    """
    Kernel class with 3 functions:
    - linear - for linear kernel 
    - rbf - for rbf kernel 
    - text - for text kernel 
    """


    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)

        return f

    @staticmethod
    def rbf(sigma=1):
        def f(x, y):
            exponent = -np.sqrt(np.linalg.norm(x - y) ** 2 / sigma)
            return np.exp(exponent)

        return f

    @staticmethod
    def text():
        def f(x, y):
            return x

        return f


def unison_shuffled_copies(a, b):
    """
    
    :param a: array a of length N
    :param b: array b of length N
    :return: shuffled arrays a and b
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class SVM:
    def __init__(self, C, kernel, epochs, rate):
        """
        
        :param C: penalty parameter C of the error term
        :param kernel: "rbf", "linear" or "text"
        :param epochs: number of iterations
        :param rate: learning rate
        """
        self.C = C
        self.kernel = kernel
        if kernel == "text":
            self._kernel = Kernel.txt()
        elif kernel == "rbf":
            self._kernel = Kernel.rbf()
        else:
            self._kernel = Kernel.linear()

        self.epochs = epochs
        self.rate = rate
        self.coef_ = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        
        :param X: training vectors
        :param y: target values where each value is either 0 or 1
        :return: Lagrange coef.
        """
        self.X = X
        y = y * 2 - 1
        self.y = y

        n, d = X.shape

        a = np.zeros(n)
        for j in range(self.epochs):
            for i in range(n):
                delta = (1 - y[i] * (np.sum(np.multiply(np.multiply(a, y), self._kernel(X, X[i])))))
                a[i] += self.rate * delta
                a[i] = min(self.C, max(0, a[i]))

        self.coef_ = a
        return a

    def get_weights(self):
        """
        :return: weights for each attribute for a trained data
        """
        n, d = self.X.shape
        weights = np.zeros(d)
        for i in range(n):
            weights += np.dot(self.coef_[i] * self.y[i], self.X[i])
        return weights

    def predict(self, X):
        """
        :param X: data in a form of a array
        :return: array of predicted classes
        """
        n, d = self.X.shape
        pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            p = 0
            for j in range(n):
                p += np.dot(self.coef_[j] * self.y[j], self._kernel(self.X[j], x))
            pred[i] = p
        return (np.sign(pred) + 1) / 2


def add_ones(X):
    """
    
    :param X: input matrix
    :return: matrix to which we add vector of ones
    """
    return np.column_stack((np.ones(len(X)), X))


def generate_data(data_type, n_samples=100):
    """
    
    :param data_type: type of data either blobs or circle
    :param n_samples: number of samples
    :return: data
    """
    np.random.seed(42)
    if data_type == "blobs":
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            centers=[[2, 2], [1, 1]],
            cluster_std=0.4
        )
    elif data_type == "circle":
        X = (np.random.rand(n_samples, 2) - 0.5) * 20
        y = (np.sqrt(np.sum(X ** 2, axis=1)) > 8).astype(int)

    X = add_ones(X)
    return X, y


def draw(X,y, errors, size, name):
    """
    
    :param X: data array
    :param y: true classes
    :param errors: indices on which we misclassified
    :param size: size of each point
    :param name: name of file to save pdf to
    """
    fig = plt.figure(1)
    ax = Axes3D(fig)

    colors = ['y' if i in errors else 'r' if el > 0 else 'b' for i, el in enumerate(y)]
    markers = ['o' if el > 0 else 'v' for el in y]
    for i, _ in enumerate(X):
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], color=colors[i], marker=markers[i], s=size[i]*10)

    r_patch = mpatches.Patch(color='r', label='Class 1')
    y_patch = mpatches.Patch(color='y', label='Misclassified data')
    b_patch = mpatches.Patch(color='b', label='Class 2')
    ax.legend(handles=[r_patch, y_patch, b_patch], loc='upper left')

    plt.axis('tight')
    plt.savefig(name, format='pdf')

if __name__ == "__main__":
    X, y = generate_data("blobs")
    svm = SVM(C=1, rate=0.001, epochs=5000, kernel="linear")
    svm.fit(X, y)
    errors = np.where(np.abs(svm.predict(X) - y) > 0)[0]
    size = svm.coef_
    draw(X,y, errors, size, 'blob.pdf')

    X, y = generate_data("circle", n_samples=200)
    svm = SVM(C=1, rate=0.001, epochs=5000, kernel="rbf")
    svm.fit(X, y)
    errors = np.where(np.abs(svm.predict(X) - y) > 0)[0]
    size = svm.coef_
    draw(X, y, errors, size, 'circle.pdf')
