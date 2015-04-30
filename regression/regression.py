__author__ = 'jamesmcnamara'
from itertools import count
from math import sqrt, exp, log
from statistics import mean
import numpy as np


from ml.utils.normalization import normalize, normalize_validation, pad


class Regressor:
    def __init__(self, design, ys, normalization="z-score", **kwargs):
        self.normalization = normalization
        self.original = design
        self.design = pad(normalize(normalization, design))
        self.ys = ys
        self.ws = np.zeros(len(self.design.T))

    def prep_test_data(self, data):
        """
            Uses the developed ws vector to make predictions for each row
            of the given data set
        :param data: a data set drawn from the same pool as the training data
        :return: a vector of y_hat predictions
        """
        return pad(normalize_validation(self.normalization,
                                        self.original, data))

    @staticmethod
    def error(design, ws, ys):
        """
            Calculates the root mean squared error obtained by using the
            given ws vector to regress the design matrix with ys as the
            actual observed values
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
    def __init__(self, design, ys, step=1e-3, tolerance=1e-3, iterations=1e5, penalty=0, **kwargs):
        super().__init__(design, ys, **kwargs)
        self.step = step
        self.tolerance = tolerance
        self.iterations = iterations
        self.penalty = penalty
        self.descent = kwargs.get("descent", True)
        self.ws = self.gradient_descent(self.design, self.ys, self.ws, descent=self.descent)

    def gradient_descent(self, design, ys, ws, step=None, tolerance=None,
                         iterations=None, descent=True):
        """
            Given a design matrix and a vector of labels, (and optionally some
            sensitivity parameters) runs the gradient descent algorithm until
            convergence within epsilon or the number of iterations is reached
        :param design: Design matrix
        :param ys: label vector
        :param step: step size for update
        :param tolerance: allowed epsilon
        :param iterations: maximum number of iterations
        :return: vector of parameters that approximately
            minimizes error function
        """
        step, tolerance, iterations = step or self.step, tolerance or self.tolerance, iterations or self.iterations
        counter = count()
        current_error = self.error(design, ws, ys)
        last_error = current_error + tolerance + 1
        print("In gradient descent")
        while next(counter) < iterations and (last_error - current_error > tolerance
                                              or current_error > last_error):
            last_error = current_error
            update = step * (self.gradient(design, ws, ys) - self.penalty * np.linalg.norm(ws))
            if descent:
                ws -= update
            else:
                ws += update
            # ws -= update if descent else -update
            current_error = self.error(design, ws, ys)
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

    @staticmethod
    def stochastic_gradient(row, ws, y):
        return row * (np.dot(ws, row) - y)


class LogisticRegressor(GradientRegressor):
    def __init__(self, design, ys, step=1e-3, tolerance=1e-3, iterations=1e5, **kwargs):
        super().__init__(design, ys, step, tolerance, iterations, descent=False, **kwargs)

    @staticmethod
    def gradient(design, ws, ys):
        """
            Calculates the gradient of the difference in predicted and actual output
            values using the given parameter vector
        :param design: observation matrix
        :param ws: a vector of parameters
        :param ys: vector observed values
        :return: a vector representing the gradient of the logistic loss function
        """
        return sum(x * (y - LogisticRegressor.prob(x, ws)) for x, y in zip(design, ys))
    
    @staticmethod
    def stochastic_gradient(row, ws, y):
        """
            Calculates the gradient of the difference in predicted and actual output
            values using the given parameter vector for only one instance
        :param design: observation matrix
        :param ws: a vector of parameters
        :param ys: vector observed values
        :return: a vector representing the gradient
        """
        return row * (LogisticRegressor.prob(row, ws) - y)

    @staticmethod
    def prob(x, ws, y=1):
        """
            Calculates the probability that the given row belongs to the
            1's class, using the given ws parameter vector
        :param x: 1*M vector consisting of one observation
        :param ws: 1*M weights vector
        :param y: label to predict, 1 or 0
        :return: The probability that x belongs to class y
        """
        return 1 / (1 + exp(-(1 if y else -1) * np.dot(ws, x)))

    @staticmethod
    def error(design, ws, ys):
        """
            Calculates the logistic loss obtained by using the
            given ws vector to regress the design matrix with ys as the
            actual observed values
        :param design: Design Matrix (N*M)
        :param ws: weight prediction vector (M*1)
        :param ys: observed results (N*1)
        :return: log loss (scalar)
        """
        predictions = (LogisticRegressor.prob(x, ws) for x in design)
        yhats = (max(1e-10, min(1-1e-10, y_hat)) for y_hat in predictions)
        return sum(y * log(y_hat) + (1-y) * log(1-y_hat) 
                   for y_hat, y in zip(yhats, ys)) 
