__author__ = 'jamesmcnamara'
from itertools import count
from math import sqrt, exp
from statistics import mean

import numpy as np


from utils.normalization import normalize, normalize_validation


class Regressor:
    def __init__(self, design, ys, normalization="z-score"):
        self.normalization = normalization
        self.original = design
        self.design = self.pad(normalize(normalization, design))
        self.ys = ys
        self.ws = np.zeros(len(self.design.T))

    def predict(self, data):
        """
            Uses the developed ws vector to make predictions for each row
            of the given data set
        :param data: a data set drawn from the same pool as the training data
        :return: a vector of y_hat predictions
        """
        return np.dot(self.pad(normalize_validation(self.normalization, self.original, data)), self.ws)

    @staticmethod
    def pad(design):
        """
            Consumes a design matrix and adds a column of 1's at the beginning
        :param design: a design matrix with N observations on M variables
        :return: a design matrix with N observations on M+1 variables
        """
        h, w = design.shape
        padded = np.ones((h, w + 1))
        padded[:, 1:] = design
        return padded

    @staticmethod
    def rmse(design, ws, ys):
        """
            Calculates the root mean squared error obtained by using the given ws vector to
            regress the design matrix with ys as the actual observed values
        :param design: Design Matrix (N*M)
        :param ws: weight prediction vector (M*1)
        :param ys: observed results (N*1)
        :return: Root mean squared error (scalar)
        """
        return sqrt(mean((np.dot(row, ws) - y) ** 2
                         for row, y in zip(design, ys)))


class NormalRegressor(Regressor):
    def __init__(self, design, ys, **kwargs):
        super().__init__(design, ys)
        if "ridge" in kwargs:
            self.ws = self.ridge_regression(self.design, self.ys, kwargs["ridge"])
        else:
            self.ws = self.normal_equations(self.design, ys)

    @staticmethod
    def normal_equations(design, ys):
        """
            Uses the hat matrix to determine the optimal regression vector
        :param design: Design matrix
        :param ys: outputs
        :return: optimal weight vector
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(design.T, design)), design.T), ys)

    @staticmethod
    def ridge_regression(design, ys, alpha):
        """
            Uses the hat matrix to determine the optimal regression vector
        :param design: Design matrix
        :param ys: outputs
        :return: optimal weight vector
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(design.T, design) + alpha * np.eye(len(design.T))), design.T), ys)


class GradientRegressor(Regressor):
    def __init__(self, design, ys, step=1e-3, tolerance=1e-3, iterations=1e5, **kwargs):
        super().__init__(design, ys, **kwargs)
        self.step = step
        self.tolerance = tolerance
        self.iterations = iterations
        self.ws = self.gradient_descent(self.design, self.ys, self.ws)

    def gradient_descent(self, design, ys, ws, step=None, tolerance=None, iterations=None, descent=True):
        """
            Given a design matrix and a vector of labels, (and optionally some sensitivity parameters)
            runs the gradient descent algorithm until convergence within epsilon or the number of
            iterations is reached
        :param design: Design matrix
        :param ys: label vector
        :param step: step size for update
        :param tolerance: allowed epsilon
        :param iterations: maximum number of iterations
        :return: vector of parameters that approximately minimizes root mean squared error
        """
        step, tolerance, iterations = step or self.step, tolerance or self.tolerance, iterations or self.iterations
        counter = count()
        current_rmse = self.rmse(design, ws, ys)
        last_rmse = current_rmse + tolerance + 1
        while next(counter) < iterations and (last_rmse - current_rmse > tolerance or current_rmse > last_rmse):

            last_rmse = current_rmse
            update = step * self.gradient(design, ws, ys)
            ws -= update if descent else -update
            current_rmse = self.rmse(design, ws, ys)
        print(ws)
        print(current_rmse)
        return ws


    @staticmethod
    def gradient(design, ws, ys):
        """
            Calculates the gradient of the difference in predicted and actual output
            values using the given parameter vector
        :param design: observation matrix
        :param ws: a vector of parameters
        :param ys: vector observed values
        :return: a vector representing the gradient
        """
        return sum(row * (np.dot(ws, row) - y) for row, y in zip(design, ys))


class LogisticRegressor(GradientRegressor):
    def __init__(self, design, ys, step=1e-3, tolerance=1e-3, iterations=1e5, **kwargs):
        super().__init__(design, ys, step, tolerance, iterations, **kwargs)

    @staticmethod
    def gradient(design, ws, ys):
        return sum(np.dot(x, y - LogisticRegressor.prob(x, ws, y=1)) for x, y in zip(design, ys))

    @staticmethod
    def prob(x, ws, y=0):
        a = exp(-np.dot(ws, x))
        num = a if y else 1
        return num / (1+a)