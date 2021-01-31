from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier


class NeuralNetwork(MLPClassifier):
    def __init__(self, hidden_layer_sizes, alpha=0.1):
        """
        Initialize NeuralNetwork classifier
        :param hidden_layer_sizes: list with numbers of hidden layer sizes
        :param alpha: float - penalty (regularization term) parameter
        """
        super().__init__(hidden_layer_sizes, alpha)
        self.coefs_ = []
        self.coefs_len = len(self.hidden_layer_sizes) + 1
        self.n_features = 0
        self.n_outputs_ = 0

        self.X = np.array([])
        self.y = np.array([])
        self.binary_y = np.array([])
        self.mask = np.array([])

    def set_data_(self, X, y):
        """
        Prepare data and other class params
        :param X: list of examples
        :param y: list of classes
        """
        # init attributes
        self.n_features = X.shape[1]
        self.n_outputs_ = len(set(y))

        self.X = X
        self.y = y
        binary_y = []
        # calculate oneHot encoding
        for cls in self.y:
            arr = [0] * self.n_outputs_
            arr[cls] = 1
            binary_y.append(arr)
        self.binary_y = np.array(binary_y)

    def fit(self, X, y):
        """
        Fit classification model with data X
        :param X: list of examples
        :param y: list of classes
        """
        self.set_data_(X, y)
        coefs_flat = self.init_weights_()

        # optimize cost function
        self.coefs_ = self.unflatten_coefs(fmin_l_bfgs_b(self.cost, x0=coefs_flat, fprime=self.grad)[0])

    def init_weights_(self):
        """
        Init weights with bias neurons and prepare for optimization process
        :return coefs: list of normalized randomly selected weights
        """
        shape_coeffs = [self.n_features] + self.hidden_layer_sizes + [self.n_outputs_]
        coefs = []
        mask = np.array([])
        for i in range(0, self.coefs_len):
            coefs.append(self.random_init(shape_coeffs[i + 1], shape_coeffs[i] + 1))
            mask_ = np.append(np.ones(shape_coeffs[i + 1]), np.zeros(shape_coeffs[i] * shape_coeffs[i + 1]))
            mask = np.append(mask, mask_)
        self.coefs_ = coefs
        coefs = self.flatten_coefs(coefs)
        self.mask = mask
        return coefs

    def random_init(self, L_in, L_out):
        """
        Prepare normalized weights for one layer
        :param L_in: int - in dimension
        :param L_out: int - out dimension
        :return weights: array of weights
        """
        init_bound = np.sqrt(6. / (L_in + L_out))
        return np.random.uniform(-init_bound, init_bound, (L_out, L_in))

    def predict(self, X):
        """
        Predict given examples classes
        :param X: list of examples
        :return classes: vector of predicted classes for given input
        """
        h = self.predict_proba(X)
        p = np.nanargmax(h, axis=1)
        return p

    def predict_proba(self, X):
        """
        Predict given examples probabilities for every class
        :param X: list of examples
        :return probs: array of predicted probabilities
        """
        A = X
        for W in self.coefs_:
            A_ones = self.add_ones(A)
            A = self.activation_fn(np.dot(A_ones, W))
        return A

    def flatten_coefs(self, coefs):
        """
        Flatten coefficients array to 1D vector
        :param coefs: coefficients array
        :return vec: coefficients vector
        """
        x = []
        for c in coefs:
            x += list(np.array(c).flatten())
        return np.array(x).flatten()

    def unflatten_coefs(self, coefs):
        """
        Unflatten coefficients 1D vector to array
        :param coefs: coefficients vector
        :return coefs: coefficients array
        """
        shape_coeffs = [self.n_features] + self.hidden_layer_sizes + [self.n_outputs_]
        c = []
        offset = 0
        for i in range(self.coefs_len):
            y_shape = shape_coeffs[i] + 1
            x_shape = shape_coeffs[i + 1]
            c.append(
                np.array(
                    [coefs[(offset + x_shape * row):(offset + x_shape * (row + 1))] for row in range(y_shape)]))
            offset += x_shape * y_shape
        return np.array(c)

    def add_ones(self, X):
        """
        Add column of ones to array
        :param X: array to add
        :return X_ones: array with ones
        """
        (m, n) = X.shape
        return np.append(np.ones((m, 1)), X, axis=1)

    def cost(self, coefs):
        """
        Cost function used for gradient descent - optimization
        :param coefs: flattened vector of coeficients
        :return J: value of cost function
        """
        u_coefs = self.unflatten_coefs(coefs)
        (m, n) = self.X.shape
        A = self.X

        for theta in u_coefs:
            A = self.add_ones(A)
            Z = np.dot(A, theta)
            A = self.activation_fn(Z)

        W = A
        J = ((W - self.binary_y) ** 2).sum()

        # regularize just non bias coefficients using mask
        J_regularization = np.sum(np.ma.array(coefs, mask=self.mask).filled(0) ** 2)
        J = (1 / (2 * m)) * J + (self.alpha / 2) * J_regularization
        return J

    def derivative_approx(self, f, x, i, delta=1e-3):
        """
        Calculate difference quotient with changing i-th value of coeffs for eps +-
        :param f: function to derive
        :param x: coeffs vector
        :param i: position of vector to vary
        :return: difference quotient vector
        """
        f1 = np.array(x)
        f2 = np.array(x)
        f1[i] += delta
        f2[i] -= delta
        return (f(f1) - f(f2)) / (2 * delta)

    def grad_approx(self, coeffs, e):
        """
        Calculate approximation for gradient
        :param coeffs: vector of coefficients
        :param e: epsilon value
        :return: difference quotient vector
        """
        return [self.derivative_approx(self.cost, coeffs, i, delta=e) for i in range(len(coeffs))]

    @staticmethod
    def activation_fn(z):
        """
        Sigmoid function
        :param z: input scalar or vector
        :return: sigmoid value of given vector or scalar
        """
        return 1 / (1 + np.exp(-z))

    def grad(self, coefs):
        """
        Gradient function used for gradient descent - optimization
        :param coeffs: flattened vector of coefficients
        :return grad: gradient value for given vector
        """
        u_coefs = self.unflatten_coefs(coefs)
        (m, n) = self.X.shape

        # front to back
        A = self.X
        Zs = []
        As = []
        grads = []
        for theta in u_coefs:
            A = self.add_ones(A)
            As.append(A)
            Z = np.dot(A, theta)
            Zs.append(Z)
            A = self.activation_fn(Z)

        delta = (A - self.binary_y) * A * (1 - A)
        for l in reversed(range(len(Zs))):
            A = self.activation_fn(Zs[l])
            # just for non last layers - for last layer we already have delta calculated
            if l + 1 < len(u_coefs):
                delta = np.dot(delta, u_coefs[l + 1].T[:, 1:]) * A * (1 - A)
            grads.append((1 / m) * np.dot(As[l].T, delta))

        # regularize non-bias coefficients using mask
        gradient = self.flatten_coefs(reversed(grads)) + self.alpha * np.ma.array(coefs, mask=self.mask).filled(0)
        return gradient


def cross_validation(X, y, k=10, compare=False):
    """
   Function to provide cross validation check (k-cross validation) on given data and compare 4 different methods
   :param X: data examples
   :param y: data classes
   :param k: number of splittings of data (k-cross) - default 10
   :param compare: parameter showing if we want to compare our method with other 3 - default False
   :return accuracy: array or scalar of prediction accuracies calculated with F1 score
   """
    set_size = len(X) // k
    remainder = len(X) % k
    results_mlp = np.zeros(len(y), dtype=int)
    results_lr = np.zeros(len(y), dtype=int)
    results_gbc = np.zeros(len(y), dtype=int)
    results_nn = np.zeros(len(y), dtype=int)
    for i in range(k):
        # split data
        learn_set = np.concatenate((X[:(i * set_size), :], X[(remainder + ((i + 1) * set_size)):, :]), axis=0)
        learn_class = y[(remainder + ((i + 1) * set_size)):]
        if i != 0:
            learn_class = np.concatenate((y[:(i * set_size)], learn_class), axis=0)
        test_set = X[(i * set_size):((i + 1) * set_size + remainder), :]
        test_indices = list(range((i * set_size), ((i + 1) * set_size + remainder)))

        nn = NeuralNetwork([30], 0.1)
        nn.fit(learn_set, learn_class)
        res = nn.predict(test_set)
        results_nn[test_indices] = res

        if compare:
            # mlp
            mlp = MLPClassifier(solver="lbfgs")
            mlp.fit(learn_set, learn_class)
            res = mlp.predict(test_set)
            results_mlp[test_indices] = res

            # logistic regression
            lr = LogisticRegression()
            lr.fit(learn_set, learn_class)
            res = lr.predict(test_set)
            results_lr[test_indices] = res
            # gradient boosting algorithm
            gbc = GradientBoostingClassifier()
            gbc.fit(learn_set, learn_class)
            res = gbc.predict(test_set)
            results_gbc[test_indices] = res

    if not compare:
        return f1_score(y, results_nn, average='micro')
    else:
        return [f1_score(y, results_nn, average='micro'), f1_score(y, results_mlp, average='micro'),
                f1_score(y, results_lr, average='micro'),
                f1_score(y, results_gbc, average='micro')]


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    scores = cross_validation(X, y, k=5, compare=True)
    for score, val in zip(["NeuralNetwork", "MLPClassifier", "LogisticRegression", "GradientBoosting"], scores):
        print("%s: %.4f " % (score, val))
