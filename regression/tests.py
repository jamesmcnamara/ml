__author__ = 'jamesmcnamara'

import numpy as np
from nose.tools import assert_equal, assert_almost_equal

from regression.regression import Regressor


class FakeDataStore:
    pass


class TestGradientRegressor:
    def setup(self):
        fds = FakeDataStore()
        fds.data = np.array([[1, 1, 2], [1, 2, 1], [1, 1, 3], [1, 3, 2]])
        fds.results = [6.4, 5.1, 8.8, 8.4]
        fds.tolerance, fds.step, fds.iterations = 1e-2, 5e-5, 10000
        fds.use_gradient = True
        self.regress = Regressor(fds.data, fds.results, fds)

    def test_gradient(self):
        predicted_grad = [-28.7, -50.6, -61.1]
        actual_grad = self.regress.gradient(self.regress.design, [0, 0, 0], self.regress.ys)
        for pred, act in zip(predicted_grad, actual_grad):
            assert_almost_equal(pred, act)

    def test_rmse(self):
        assert_almost_equal(self.regress.rmse(self.regress.design, [0, 0, 0], self.regress.ys), 7.33, delta=0.01)
        assert_almost_equal(self.regress.rmse(self.regress.design, [1, 1, 2], self.regress.ys), 0.5, delta=0.01)

    def test_gradient_descent(self):
        reg = self.regress.gradient_descent(self.regress.design, [6, 5, 8, 8], [0, 0, 0], 1e-2, 1e-6, 10000)
        for actual, expected in zip(reg, [1, 1, 2]):
            assert_almost_equal(actual, expected, delta=0.01)

    def test_predict(self):
        assert_almost_equal(self.regress.predict([[1, 7, 4]])[0], 17, delta=0.5)
        assert_almost_equal(self.regress.predict([[1, 2, 3]])[0], 9.5, delta=0.5)

    def test_pad(self):
        design = np.zeros((3, 2))
        output = np.array([[1, 0, 0] for _ in range(3)])
        for pad_row, out_row in zip(self.regress.pad(design), output):
            assert_equal(list(pad_row), list(out_row))