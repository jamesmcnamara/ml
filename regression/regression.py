__author__ = 'jamesmcnamara'
from itertools import count
from math import sqrt

import numpy as np


class Regressor:
    def __init__(self, design, ys, datastore, **kwargs):
        self.design = self.pad(design)
        self.ys = ys

        self.datastore = datastore
        self.ws = np.zeros(len(self.design.T))
        if datastore.use_gradient:
            self.step = datastore.step
            self.tolerance = datastore.tolerance
            self.iterations = datastore.iterations
            self.collect_rmse = datastore.collect_rmse
            self.rmses = []
            self.ws = self.gradient_descent(self.design, self.ys, self.ws)
        elif "ridge" in kwargs:
            self.ws = self.ridge_regression(self.design, self.ys, kwargs["ridge"])
        else:
            self.ws = self.normal_equations(self.design, ys)

    def gradient_descent(self, design, ys, ws, step=None, tolerance=None, iterations=None):
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
        last_rmse = tolerance
        while next(counter) < iterations and last_rmse - self.rmse(design, ws, ys) > tolerance:
            last_rmse = self.rmse(design, ws, ys)
            ws -= step * self.gradient(design, ws, ys)
            if self.collect_rmse:
                self.rmses.append(self.rmse(design, ws, ys))
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

    def predict(self, data):
        """
            Uses the developed ws vector to make predictions for each row
            of the given data set
        :param data: a data set drawn from the same pool as the training data
        :return: a vector of y_hat predictions
        """

        return [np.dot(self.ws, row) for row in self.pad(data)]

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
    def normal_equations(design, ys):
        """
            Uses the hat matrix to determine the optimal regression vector
        :param design: Design matrix
        :param ys: outputs
        :return: optimal weight vector
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(design.T, design)), design.T), ys)

    @staticmethod
    def ridge_regression(design, ys, lmbda):
        """
            Uses the hat matrix to determine the optimal regression vector
        :param design: Design matrix
        :param ys: outputs
        :return: optimal weight vector
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(design.T, design) + lmbda * np.eye(len(design.T))), design.T), ys)


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
        return sqrt(sum((np.dot(ws, row) - y) ** 2 for row, y in zip(design, ys)) / len(design))

