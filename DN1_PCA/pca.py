import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import decomposition


class EigenPCA:
    """
    Calculates full PCA transformation by using Numpy's np.linalg.eigh.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Fits model according to the X.
        Calculates self.components_, self.explained_variance_ and self.explained_variance_ratio
        """

        # 1. center data
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_

        # 2. calculate covariance matrix S
        S = X.transpose().dot(X)

        # 3. calculate eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        # flip columns, because eigenvectors must be based on the descending eigenvalues
        self.components_ = np.flip(eigenvectors, axis=1).transpose()[: self.n_components]
        trace = sum(eigenvalues)
        eigenvalues = np.array(list(reversed(eigenvalues)))[: self.n_components]
        self.explained_variance_ = eigenvalues / X.shape[0]
        self.explained_variance_ratio_ = eigenvalues / trace
        return self

    def transform(self, X):
        """
        Returns X projected to n_components
        """
        X = X - self.mean_
        return self.components_.dot(X.transpose()).transpose()


class PowerPCA:
    """
    Calculates full PCA transformation by using Powers method.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.eps = 1e-10
        self.max_iters = 1000
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Fits model according to the X.
        Calculates self.components_, self.explained_variance_ and self.explained_variance_ratio
        """

        calculated_components = 0
        self.components_ = []
        eigenvalues = []
        trace = 0

        # center data
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_

        # calculate n_components eigenvectors
        while calculated_components < self.n_components:
            # calculate covariance matrix S
            S = X.transpose().dot(X)
            # in the first iteration calculate matrix trace
            if calculated_components == 0:
                trace = np.trace(S)
            # create random vector with size = no. attributes
            u = np.random.rand(X.shape[1])

            iter = 0
            # calculate one eigenvector
            while True:
                # change vectors direction
                u_ = S.dot(u)
                # normalize
                u_ = u_ / np.linalg.norm(u_)
                diff = np.linalg.norm(u_ - u)
                u = u_

                if diff < self.eps or iter > self.max_iters:
                    break
                iter += 1

            self.components_.append(u)
            # from X remove projections on the calculated eigenvector u
            X = X - np.outer(X.dot(u), u)
            eigenval = u.dot(S).dot(u)
            eigenvalues.append(eigenval)
            calculated_components += 1

        self.components_ = np.array(self.components_)
        # calculate explained_variance_ and explained_variance_ratio_
        eigenvalues = np.array(eigenvalues)
        self.explained_variance_ = eigenvalues / X.shape[0]
        self.explained_variance_ratio_ = eigenvalues / trace
        return self

    def transform(self, X):
        """
        Returns X projected to n_components
        """
        X = X - self.mean_
        return self.components_.dot(X.transpose()).transpose()


class OrtoPCA:
    """
    Calculates full PCA transformation by using Powers method and Gram-Schmidt Orthonormalization.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.eps = 1e-10
        self.max_iters = 1000
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Fits model according to the X.
        Calculates self.components_, self.explained_variance_ and self.explained_variance_ratio
        """

        # center data
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_

        # calculate covariance matrix S
        S = X.transpose().dot(X)

        trace = np.trace(S)

        # create matrix of random vectors
        U = np.random.rand(self.n_components, X.shape[1])

        iter = 0
        # calculate eigenvectors
        while True:
            # change vectors direction
            U_ = S.dot(U.transpose()).transpose()

            # make U orthogonal
            U_ = self.gram_schmidt(U_)

            diff = np.linalg.norm(U_ - U)
            U = U_

            if diff < self.eps or iter > self.max_iters:
                break
            iter += 1

        self.components_ = U

        # calculate eigenvalues, explained_variance_ and explained_variance_ratio_
        eigenvalues = U.dot(S).dot(U.transpose()).diagonal()
        self.explained_variance_ = eigenvalues / X.shape[0]
        self.explained_variance_ratio_ = eigenvalues / trace
        return self

    def transform(self, X):
        """
        Returns X projected to n_components
        """
        X = X - self.mean_
        return self.components_.dot(X.transpose()).transpose()

    def gram_schmidt(self, U):
        """
        Performs Gram-Schmidt Orthonormalization on the matrix U
        """

        # 1. make projections
        for i in range(len(U)):
            if i != 0:
                projections = 0
                for j in reversed(range(0, i)):
                    projection = (U[j].dot(U[i]) / U[j].dot(U[j])) * U[j]
                    projections -= projection
                U[i] += projections

        # 2. normalize U
        for u in U:
            u /= np.linalg.norm(u)
        return U


def plot_data(Z, numbers):
    print("# Plotting data...")
    f = plt.figure()
    data = {}
    averages = {}
    for i in range(Z.shape[0]):
        no = numbers[i]
        if no not in data:
            data[no] = Z[i]
            averages[no] = Z[i]
        else:
            data[no] = np.vstack((data[no], Z[i]))
            averages[no] += Z[i]

    # plot data
    for n in range(0, 10):
        plt.scatter(data[n][:, 0], data[n][:, 1], 4, label=str(n))

    # plot numbers
    for n in range(0, 10):
        plt.text(averages[n][0] / data[n].shape[0], averages[n][1] / data[n].shape[0], n, fontsize=15,
                 bbox={'facecolor': 'white', 'alpha': 1, 'pad': 2})

    plt.title("PCA")
    plt.legend()

    # show chart
    plt.show()

    # save to the file
    f.savefig('pca.pdf')
    print("# Data plotted.")


def load_data(filename):
    print("# Loading data...")
    f = open(filename, "rt")
    f.readline()  # skip first row
    X = []
    numbers = []
    for l in csv.reader(f):
        X.append(l[1:])
        numbers.append(l[0])

    print("# Data loaded.")
    return np.array(X).astype(np.int), np.array(numbers).astype(np.int)


if __name__ == "__main__":
    n_comp = 2
    test_data_percentage = 0.95

    # read data from file
    X, numbers = load_data('train.csv')

    # divide data on train and test set
    test_len = round(len(X) * test_data_percentage)
    train, test = X[:test_len], X[(len(X) - test_len):]

    # perform PCA
    my_pca = OrtoPCA(n_comp)
    # my_pca = PowerPCA(n_comp)
    # my_pca = EigenPCA(n_comp)

    my_pca.fit(train)
    Z = my_pca.transform(test)

    # plot
    plot_data(Z, numbers[(len(X) - test_len):])
