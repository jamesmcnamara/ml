__author__ = 'jamesmcnamara'

from math import sqrt

import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from sklearn.linear_model import Ridge

from learn import RegressionDataStore
from regression.regression import Regressor


class FakeDataStore:
    def __init__(self):
        self.data = np.array([[1, 1, 2], [1, 2, 1], [1, 1, 3], [1, 3, 2]])
        self.results = np.array([6.4, 5.1, 8.8, 8.4])
        self.tolerance, self.step, self.iterations = 1e-2, 5e-5, 10000
        self.use_gradient = True
        self.collect_rmse = False


class FakeParser:
    def __init__(self, infile, meta=None, cross=10, normalization="z-score", powers=1, validation=None, debug=False):
        self.infile = infile
        self.meta = meta or open(infile.name.replace(".csv", ".meta"))
        self.cross = cross
        self.normalization = normalization
        self.powers = powers
        self.validation = validation
        self.debug = debug
        self.plot_rmse = False
        self.gradient_descent = False


class TestGradientRegressor:
    def setup(self):
        fds = FakeDataStore()
        self.regress = Regressor(fds.data, fds.results, fds)

    # def test_gradient(self):
    #     predicted_grad = [-28.7, -50.6, -61.1]
    #     actual_grad = self.regress.gradient(self.regress.design, [0, 0, 0], self.regress.ys)
    #     for pred, act in zip(predicted_grad, actual_grad):
    #         assert_almost_equal(pred, act)

    # def test_rmse(self):
    #     assert_almost_equal(self.regress.rmse(self.regress.design, [0, 0, 0, 0], self.regress.ys), 7.33, delta=0.01)
    #     assert_almost_equal(self.regress.rmse(self.regress.design, [1, 1, 2, 0], self.regress.ys), 0.5, delta=0.01)
    #
    # def test_gradient_descent(self):
    #     reg = self.regress.gradient_descent(self.regress.design, [6, 5, 8, 8], [0, 0, 0, 0], 1e-3, 1e-5, 10000)
    #     for actual, expected in zip(reg, [1, 1, 2]):
    #         assert_almost_equal(sqrt(actual), expected, delta=0.01)

    # def test_predict(self):
    #     assert_almost_equal(self.regress.predict(np.array([[1, 7, 4]]))[0], 17, delta=0.5)
    #     assert_almost_equal(self.regress.predict(np.array([[1, 2, 3]]))[0], 9.5, delta=0.5)

    def test_pad(self):
        design = np.zeros((3, 2))
        output = np.array([[1, 0, 0] for _ in range(3)])
        for pad_row, out_row in zip(self.regress.pad(design), output):
            assert_equal(list(pad_row), list(out_row))

    def test_normal_equations(self):
        rds = RegressionDataStore(FakeParser(open("data/housing.csv")))
        reg = Regressor(rds.data, rds.results, rds)
        clf = Ridge(alpha=0, fit_intercept=False)
        clf.fit(reg.design, reg.ys)
        for actual, expected in zip(reg.normal_equations(reg.design, reg.ys), clf.coef_):
            assert_almost_equal(actual, expected, delta=0.01)

    def test_ridge_regression(self):
        rds = RegressionDataStore(FakeParser(open("data/housing.csv")))
        reg = Regressor(rds.data, rds.results, rds)
        for i in range(10):
            clf = Ridge(alpha=i/10, fit_intercept=False)
            clf.fit(reg.design, reg.ys)
            for actual, expected in zip(reg.ridge_regression(reg.design, reg.ys, i/10), clf.coef_):
                assert_almost_equal(actual, expected, delta=0.01)
