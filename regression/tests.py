__author__ = 'jamesmcnamara'

from math import sqrt
from random import random

import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from sklearn.linear_model import Ridge

from learn import RegressionDataStore
from regression.regression import NormalRegressor, GradientRegressor


class FakeDataStore:
    def __init__(self):
        self.data = np.array([[1, 1, 2], [1, 2, 1], [1, 1, 3], [1, 3, 2]])
        self.results = np.array([6.4, 5.1, 8.8, 8.4])
        self.tolerance, self.step, self.iterations = 1e-2, 5e-5, 10000
        self.use_gradient = True
        self.collect_rmse = False


class FakeParser:
    def __init__(self, infile, meta=None, cross=10, normalization="z-score",
                 powers=1, validation=None, debug=False):
        self.infile = infile
        self.meta = meta or open(infile.name.replace(".csv", ".meta"))
        self.cross = cross
        self.normalization = normalization
        self.powers = powers
        self.validation = validation
        self.debug = debug
        self.plot_rmse = False
        self.gradient_descent = False


class TestRegressor:
    def setup(self):
        rds = RegressionDataStore(FakeParser(open("data/housing.csv")))
        self.norm = NormalRegressor(rds.data, rds.results)
        self.grad = GradientRegressor(rds.data, rds.results)
        self.fds = FakeDataStore()
        self.default_design = self.grad.pad(self.fds.data)
        
    def test_gradient(self):
        predicted_grad = [-28.7, -50.6, -61.1]
        actual_grad = self.grad.gradient(self.fds.data, np.array([0, 0, 0]),
                                         self.fds.results)
        for pred, act in zip(predicted_grad, actual_grad):
            assert_almost_equal(pred, act)

    def test_rmse(self):
        assert_almost_equal(self.grad.rmse(self.default_design,
                                           [0, 0, 0, 0], self.fds.results),
                            7.33, delta=0.01)
        assert_almost_equal(self.grad.rmse(self.default_design,
                                           [0, 1, 1, 2], self.fds.results),
                            0.5, delta=0.01)
    
    def test_gradient_descent(self):
        ws = np.array([1, 1, 1, 2])
        mat = np.array([[1] + [random() * 20 for _ in range(3)] for _ in range(100)])
        mat *= ws
        output = np.array(np.dot(mat, ws))
        reg = self.grad.gradient_descent(mat, output,
                                         [1, 0, 0, 0], step=1e-5, tolerance=1e-6, iterations=1e5)
        # for row, result in zip(self.default_design, [6, 5, 8, 8]):
        #     print(np.dot(row, reg) - result)
        for actual, expected in zip(reg, ws):
            assert_almost_equal(actual, expected, delta=0.1)

    def test_predict(self):
        self.grad.ws = np.array([0, 1, 1, 2])
        predictions = self.grad.predict(np.array([[1, 7, 4], [1, 2, 3]]))
        expected = [16, 9.5]
        for act, exp in zip(predictions, expected):
            assert_almost_equal(act, exp, delta=0.5)

    def test_pad(self):
        design = np.zeros((3, 2))
        output = np.array([[1, 0, 0] for _ in range(3)])
        for pad_row, out_row in zip(NormalRegressor.pad(design), output):
            assert_equal(list(pad_row), list(out_row))

    def test_normal_equations(self):
        clf = Ridge(alpha=0, fit_intercept=False)
        clf.fit(self.norm.design, self.norm.ys)
        for actual, expected in zip(self.norm.normal_equations(self.norm.design,
                                                               self.norm.ys),
                                    clf.coef_):
            assert_almost_equal(actual, expected, delta=0.01)

    def test_ridge_regression(self):
        for i in range(10):
            clf = Ridge(alpha=i/10, fit_intercept=False)
            clf.fit(self.norm.design, self.norm.ys)
            for actual, expected in \
                    zip(self.norm.ridge_regression(self.norm.design,
                                                   self.norm.ys, i/10),
                        clf.coef_):
                assert_almost_equal(actual, expected, delta=0.01)
