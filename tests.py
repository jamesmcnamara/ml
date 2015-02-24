from io import StringIO
import json
from math import sqrt

from nose.tools import assert_equal, assert_greater, assert_almost_equal
import numpy as np

from learn import DecisionTreeDataStore
__author__ = 'jamesmcnamara'


def get_meta():
    return StringIO(json.dumps(dict(name="iris", height=15, width=5, data_type="float", result_type="str")))


def get_data():
    data = ['5.0,3.4,1.6,0.4,Iris-setosa',
            '5.1,3.7,1.5,0.4,Iris-setosa',
            '5.2,2.7,3.9,1.4,Iris-versicolor',
            '4.8,3.0,1.4,0.3,Iris-setosa',
            '6.7,3.0,5.2,2.3,Iris-virginica',
            '4.6,3.4,1.4,0.3,Iris-setosa',
            '5.0,3.0,1.6,0.2,Iris-setosa',
            '6.4,2.7,5.3,1.9,Iris-virginica',
            '5.0,3.2,1.2,0.2,Iris-setosa',
            '5.5,3.5,1.3,0.2,Iris-setosa',
            '5.4,3.9,1.7,0.4,Iris-setosa',
            '4.5,2.3,1.3,0.3,Iris-setosa',
            '4.4,2.9,1.4,0.2,Iris-setosa',
            '5.6,2.7,4.2,1.3,Iris-versicolor',
            '6.3,3.4,5.6,2.4,Iris-virginica']

    return StringIO("\n".join(data))


class FakeParser:
    pass


class TestDataStore:

    def setup(self):
        fake_parser = FakeParser()
        fake_parser.meta = get_meta()
        fake_parser.infile = get_data()
        fake_parser.cross = 10
        fake_parser.tree = "entropy"
        fake_parser.range = (5, 25, 5)
        fake_parser.debug = False
        fake_parser.with_confusion = False
        fake_parser.binary_splits = False
        fake_parser.normalization = "arithmetic"
        fake_parser.validation = None
        fake_parser.powers = None
        self.ds = DecisionTreeDataStore(fake_parser)

    def teardown(self):
        del self.ds

    def iris_setup(self):
        fake_parser = FakeParser()
        fake_parser.meta = open("data/iris.meta")
        fake_parser.infile = open("data/iris.csv")
        fake_parser.cross = 10
        fake_parser.tree = "entropy"
        fake_parser.range = (5, 25, 5)
        fake_parser.debug = False
        fake_parser.with_confusion = False
        fake_parser.binary_splits = False
        fake_parser.normalization = "arithmetic"
        fake_parser.validation = None
        fake_parser.powers = None
        return DecisionTreeDataStore(fake_parser)

    def test_extract(self):
        data, results = self.ds.extract(self.ds.data_type, self.ds.result_type,
                                        self.ds.width, self.ds.height, get_data())
        assert_equal(type(data), np.ndarray)
        assert_equal(len(data), 15)
        assert_equal(len(data[0]), 4)
        assert_equal(type(data[0][0]), np.float64)
        assert_equal(len(results), 15)
        assert_equal(type(results[0]), str)
        assert_equal(int(sum(sum(data))), 177)

    def test_normalize_arithmetic(self):
        data = self.ds.normalize_columns_arithmetic(self.ds.data)
        for column in data.T:
            assert_equal(max(column), 1)
            assert_equal(min(column), 0)
        first_col = self.ds.data[:, 0]
        first_col_norm = data[:, 0]
        top, bot = max(first_col), min(first_col)
        for orig, norm in zip(first_col, first_col_norm):
            assert_equal((orig - bot) / (top - bot), norm)

    def test_normalize_z(self):
        ds = self.iris_setup()
        data = ds.normalize_columns_z(ds.data)
        for column in data.T:
            avg = sum(column) / len(column)
            sd = sqrt(sum(map(lambda elem: (elem - avg) ** 2, column)) / (len(column) - 1))
            assert_almost_equal(avg, 0)
            assert_almost_equal(sd, 1)

    def test_normalize_z_validation(self):
        training = np.array([[i % 2 for i in range(100)], [i % 4 for i in range(100)]]).T
        normed_ones = DecisionTreeDataStore.normalize_columns_z_validation(training, np.array([[1 for _ in range(10)],
                                                                                               [2 for _ in range(10)]]).T)
        for row in normed_ones:
            one, two = row
            assert_almost_equal(one, 1, delta=0.1)
            assert_almost_equal(two, 0.5, delta=0.1)

    def test_cross_validation(self):
        ds = self.iris_setup()
        for eta in range(5, 26, 5):
            accuracies = ds.cross_validation(ds.accuracy, ds.data, ds.results, ds.k_validation, eta)
            avg = sum(accuracies) / len(accuracies)
            sd = DecisionTreeDataStore.sample_sd(accuracies)
            assert_greater(avg, 0.80)
            assert_greater(1, avg)
            assert_greater(0.15, sd)
            assert_greater(sd, 0)

    def test_add_powers(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        output = self.ds.add_powers(data, 2)

        squared = np.array([[1, 2, 3, 1, 4, 9], [4, 5, 6, 16, 25, 36], [7, 8, 9, 49, 64, 81]])
        for out_row, expected_row in zip(output, squared):
            for out_val, expected_val in zip(out_row, expected_row):
                assert_equal(out_val, expected_val)

        data = np.array([[2, 3], [3, 4]])
        output = self.ds.add_powers(data, 3)
        cubed = np.array([[2, 3, 4, 9, 8, 27], [3, 4, 9, 16, 27, 64]])

        for out_row, expected_row in zip(output, cubed):
            for out_val, expected_val in zip(out_row, expected_row):
                assert_equal(out_val, expected_val)

