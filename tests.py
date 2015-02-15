from io import StringIO
import json

from nose.tools import assert_equals, assert_greater
import numpy as np
from mock import patch

from learn import DataStore
__author__ = 'jamesmcnamara'


def get_meta():
    return StringIO(json.dumps(dict(name="iris", height=15, width=5, type="float")))


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


class TestDataStore:

    @patch("learn.load_args")
    def setup(self, load_args_patch):
        instance = load_args_patch.return_value
        instance.meta = get_meta()
        instance.infile = get_data()
        instance.cross = 10
        instance.tree = "entropy"
        instance.range = (5, 25, 5)
        instance.debug = False
        instance.with_confusion = False
        instance.binary_splits = False
        self.ds = DataStore()

    @patch("learn.load_args")
    def iris_setup(self, load_args_patch):
        instance = load_args_patch.return_value
        instance.meta = open("data/iris.meta")
        instance.infile = open("data/iris.csv")
        instance.cross = 10
        instance.tree = "entropy"
        instance.range = (5, 25, 5)
        instance.debug = False
        instance.with_confusion = False
        instance.binary_splits = False
        return DataStore()

    def test_extract(self):
        data, results = self.ds.extract(self.ds.type, self.ds.width, self.ds.height, get_data())
        assert_equals(type(data), np.ndarray)
        assert_equals(len(data), 15)
        assert_equals(len(data[0]), 4)
        assert_equals(type(data[0][0]), np.float64)
        assert_equals(len(results), 15)
        assert_equals(type(results[0]), str)
        assert_equals(int(sum(sum(data))), 177)

    def test_normalize(self):
        data = self.ds.normalize_columns(self.ds.data)
        for column in data.T:
            assert_equals(max(column), 1)
            assert_equals(min(column), 0)

    def test_cross_validation(self):
        self.ds = self.iris_setup()
        for eta in range(5, 26, 5):
            avg, sd = self.ds.cross_validation(5, self.ds.accuracy)
            assert_greater(avg, 0.85)
            assert_greater(1, avg)
            assert_greater(0.15, sd)
            assert_greater(sd, 0)



