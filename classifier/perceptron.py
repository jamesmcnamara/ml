from itertools import count
from math import exp

import numpy as np
from ml.utils.normalization import pad
__author__ = 'jamesmcnamara'


class Perceptron:
    def __init__(self, data=None, labels=None, step=1e-4, iterations=1e5):
        self.data = pad(data)
        self.labels = labels
        self.step = step
        self.ws = self.find_weight_vector(self.data, labels, step, iterations) 

    def predict(self, data):
        partial_classify = lambda row: np.sign(self.classify(row, self.ws))
        print(self.ws)
        return map(partial_classify, pad(data))
    
    @staticmethod
    def find_weight_vector(data, labels, step, iterations):
        """
            Searches for a weight vector which linearly separates data
            into the classes of labels
        :param data: Input data N*M matrix
        :param labels: N*1 vector of associated labels, -1 or 1
        :param step: step size, float
        :param iterations: max number of steps, int
        :return: weight vector that separates data, or most nearly separates
            when max iterations were reached
        """
        c = count()
        ws = np.zeros(len(data.T))
        error = True

        while error and next(c) < iterations:
            error = False
            for row, y in zip(data, labels):
                y_hat = Perceptron.classify(row, ws)
                if y * y_hat <= 0:
                    ws += step * row * y
                    error = True
        return ws
    
    @staticmethod
    def classify(row, ws):
        """
            Classifies the given row using the given ws
        :param row: N*1 observation vector
        :param ws: N*1 weight vector
        :return: classification float, sign denotes classification
        """
        return np.dot(row, ws)


class DualPerceptron:
    def __init__(self, data=None, labels=None, step=1e-4, iterations=1e5,
                 kernel="linear", gamma=1):
        self.data = pad(data)
        self.labels = labels
        if kernel == "rbf":
            self.gamma, self.alphas = self.search_gamma_alphas(step,
                                                               iterations)
            self.kernel = self.radial_basis_generator(gamma)
        else:
            self.alphas = self.find_weight_vector(self.data, labels, 
                                                  iterations,
                                                  self.linear_kernel)
            self.kernel = self.linear_kernel
        
    def predict(self, data, kernel_function=None, add_pad=True):
        """
            returns a stream of predictions for the given data
        :param data: N*M data matrix
        :param kernel_function: a function to kernelize over,
            default linear kernel
        :return: stream of predictions (N*1)
        """
        kernel = kernel_function or self.kernel

        def partial_classify(row):
            """
                Creates a closure that can classify a given row using
                the current state of the classifier
                :param row: a given test data row (1*M vector)
                :return: prediction, int {0, 1}
            """
            return np.sign(self.classify(row, self.alphas, self.data, 
                                         self.labels, kernel))
        return map(partial_classify, pad(data) if add_pad else data)

    def search_gamma_alphas(self, step, iterations):
        """
            Attempts to find the best combination of gamma and alpha weights
            for the RBF kernel
        :param step: step size to update weights by
        :param iterations: max number of iterations to search before exiting
            and returning the best result so far
        :return: gamma float and alphas M*1 vector
        """
        gamma = 0
        err = True
        while err:
            print("err with", gamma)
            gamma += step
            kernel = self.radial_basis_generator(gamma)
            self.alphas = self.find_weight_vector(self.data, self.labels, 
                                                  iterations, kernel)
            err = not all(pred == act for pred, act in 
                          zip(self.predict(self.data, kernel_function=kernel,
                                           add_pad=False),
                              self.labels))
        return gamma, self.alphas
    
    @staticmethod
    def radial_basis_generator(gamma):
        """
            returns an RBF kernel with the given gamma parameter
        :param gamma: gamma parameter of the RBF kernel
        :return: RBF kernel function
        """
        return lambda x, x_i: exp(-gamma * np.linalg.norm(x - x_i)**2)

    @staticmethod
    def linear_kernel(x, x_i):
        """
            Linear kernel, default. Simply a dot product
        :param x: observation from the input data (M*1)
        :param x_i: observation that we are classifying (M*1)
        :return: classification of x_i, float
        """
        return np.dot(x, x_i)
    
    @staticmethod
    def classify(row, alphas, data, labels, kernel):
        """
            Classifies the given row using the alphas weight vector,
            input data, associated labels, and a kernel
        :param row: M*1 vector to classify as 1 or -1
        :param alphas: M*1 weight vector
        :param data: N*M input data matrix
        :param labels:  N*1 labels data
        :param kernel: Kernel function (N*1 x N*1 -> float)
        :return: classification for row
        """
        return sum(alpha_i * y_i * kernel(row_i, row) 
                   for alpha_i, row_i, y_i in 
                   zip(alphas, data, labels))
    
    @staticmethod
    def find_weight_vector(data, labels, iterations, kernel):
        """
            finds the best set of weights separating the given data into the given
            labels using the given kernel function
        :param data: N*M data matrix
        :param labels: N*1 set of associated labels (-1 or 1)
        :param iterations: maximum number of iterations, int
        :param kernel: Kernel function (N*1 x N*1 -> float)
        :return: N*1 weight vector for classifying the given data
        """
        alphas = np.zeros(len(data))
        error = True
        c = count()
        while error and next(c) < iterations:
            error = False
            for i, (row, y) in enumerate(zip(data, labels)):
                y_hat = DualPerceptron.classify(row, alphas, data, 
                                                labels, kernel)
                if y * y_hat <= 0:
                    alphas[i] += 1
                    error = True
        return alphas
