from collections import namedtuple
from itertools import compress
from math import sqrt
from operator import eq


import argparse
import json
import numpy as np

from dtree.classification_tree import EntropyTree, CategoricalEntropyTree
from dtree.regression_tree import RegressionTree

__author__ = 'jamesmcnamara'

tree_map = {"entropy": EntropyTree, "regression": RegressionTree, 'categorical': CategoricalEntropyTree}


def load_args():
    """
        Parses the command line arguments, and loads the arguments onto an ArgumentParser object, which it returns
    :return: an ArgumentParser object with all of the specified command line features written on
    """
    parser = argparse.ArgumentParser(description="General purpose command line machine learning tool.")

    parser.add_argument("infile", type=open, help="CSV file with training data")
    parser.add_argument("-r", "--range", type=int, nargs=3,
                        help="Range of eta values to use for cross validation. The first "
                             "value is start, the second is end, and the last is interval")
    parser.add_argument("-m", "--meta", type=open, default=None,
                        help="Meta file containing JSON formatted descriptions of the data")

    parser.add_argument("-cv", "--cross", type=int, default=10,
                        help="Set the parameter for k-fold cross validation. Default 10")
    parser.add_argument("-t", "--tree", type=str, default="entropy",
                        help="What type of decision tree to build for the data. Options are "
                             "'entropy', 'regression', or 'categorical'. Default 'entropy'")
    parser.add_argument("-d", "--debug", action="store_true", default=False,
                        help="Use sci-kit learn instead of learn.py, to test that the behavior is correct")

    parser.add_argument("-cf", "--with-confusion", action="store_true", default=False,
                        help="Include a confusion matrix in the output")

    parser.add_argument("-b", "--binary-splits", action="store_true", default=False,
                        help="Convert a multi-way categorical matrix to a binary matrix")

    return parser.parse_args()


class DataStore:
    def __init__(self):
        # Load data
        parser = load_args()
        self.meta = json.load(parser.meta) if parser.meta \
            else json.load(open(parser.infile.name.replace(".csv", ".meta")))
        self.width, self.height, self.type = self.meta["width"], self.meta["height"], self.meta["type"]
        self.data, self.results = self.extract(self.type, self.width, self.height, parser.infile)
        self.k_validation = parser.cross
        self.tree_type = tree_map[parser.tree]
        self.confusion = parser.with_confusion
        self.debug = parser.debug
        self.start, self.stop, self.step = parser.range

        # Clean Data
        self.shuffle()
        if parser.binary_splits:
            self.data = self.binarize_columns(self.data)
            self.tree_type = tree_map["entropy"]

        # Set up testing functions
        if self.debug:
            self.test_func = self.scikit_classifier if self.tree_type == EntropyTree else self.scikit_regressor
        else:
            self.test_func = self.accuracy

        # Reference data for classifiers
        if self.tree_type != RegressionTree:
            self.result_types = set(self.results)
            self.result_map = {result: i for i, result in enumerate(self.result_types)}
            self.results = [self.result_map[val] for val in self.results]
        else:
            self.results = list(map(float, self.results))


    @staticmethod
    def extract(type, width, height, file):
        """
            Extracts and loads the data from the given file into a numpy matrix and the
            results into a numpy array
        :param file: input CSV file with the first n-1 columns containing the observations and the
            last column containing the results from the training set
        :return: a tuple of the data in matrix form and an array of the result data
        """
        transform = lambda values: list(map(float if type == "float" else str, values))
        data = np.zeros((height, width - 1), type)
        results = [""] * height
        for i, line in enumerate(file):
            *xs, y = line.split(",")
            data[i] = transform(xs)
            results[i] = y.strip("\n")
        return data, results

    def shuffle(self):
        """
            Randomly shuffles the input data matrix to allow for random sampling
        """
        seed = np.random.randint(0, 100000)
        np.random.seed(seed)
        np.random.shuffle(self.data)
        np.random.seed(seed)
        np.random.shuffle(self.results)
        np.random.seed(np.random.randint(0, 100000))

    @staticmethod
    def normalize_columns(data):
        """
            For all columns with numeric type, normalize their values to be between [0,1]
        """
        # Helper function which maps the normalization function over a column from the input matrix
        def column_mapper(col):
            minimum, maximum = min(col), max(col)
            return list(map(lambda x: (x - minimum) / (maximum - minimum), col))
        return np.array([column_mapper(col) for col in data.T]).T


    @staticmethod
    def binarize_columns(data):
        """
            Consumes a matrix of categorical features and translates it into a binary matrix of
        :param data:
        :return:
        """
        # Offset data container
        OffsetData = namedtuple("ColData", ["offset", "symbol_map"])
        offset, cols = 0, []

        # Create a list where the i-th element contains the offset for the i-th column and a dictionary mapping
        # each possible symbol in ith column to its additional offset
        for column in data.T:
            cols.append(OffsetData(offset, {symbol: i for i, symbol in enumerate(set(column))}))
            offset += len(cols[-1].symbol_map)

        last_offset, last_symbols = cols[-1]
        width = last_offset + len(last_symbols)

        # Initialize the array of binary data
        bin_data = np.zeros((len(data), width), np.dtype('b'))
        for i, row in enumerate(data):
            binary_row = np.zeros(width, np.dtype('b'))
            for column, symbol in enumerate(row):
                offset_data = cols[column]
                binary_row[offset_data.offset + offset_data.symbol_map[symbol]] = True
            bin_data[i] = binary_row
        return bin_data

    def test(self, start, stop, step, accuracy_func):
        """
            Consumes descriptors of various eta levels and an accuracy function and determines
            prints messages about the average accuracy and variation across the folds
        :param start: first eta min value to test
        :param stop: last eta min value to test
        :param step: strides for updating eta mins from start to stop
        :param accuracy_func: a function which determines average accuracy k constructed tree with
            the given eta min using cross fold validation, where k is the k validation parameter
        """
        for eta_percent in range(start, stop + 1, step):
            avg, sd = self.cross_validation(eta_percent, accuracy_func)

            if self.tree_type != RegressionTree:
                print("{}% eta gave a classification accuracy of {:.2f}% with a standard deviation of {:.2f}%"
                      .format(eta_percent, avg * 100, sd * 100))
            else:
                print("{}% eta had an average MSE of {:.2f} with a standard deviation of {:.2f}%"
                      .format(eta_percent, avg, sd))
    @staticmethod
    def chunk(data, segments):
        """
            Consumes a list and a number and returns a the input list split into n chunks in order
        """
        return list(zip(*[iter(data)] * (len(data) // segments)))

    def cross_validation(self, eta, accuracy_func):
        """
            Splits this data set into k chunks based on the k_validation attribute, and excludes each chunk,
            one at a time, from the input data set to a decision tree, and then tests that decision tree on
            the excluded data set. The function accumulates each of the accuracy measures, and returns the
            average accuracy and the standard deviation
        :param eta: eta min for the decision trees (early stopping strategy)
        :param accuracy_func: a function which determines average accuracy k constructed tree with
            the given eta min using cross fold validation, where k is the k validation parameter
        :return: 2-tuple of average accuracy and standard deviation
        """

        data_chunks = self.chunk(self.data, self.k_validation)
        result_chunks = self.chunk(self.results, self.k_validation)
        accuracies = [self.get_ith_accuracy(data_chunks, result_chunks, eta, accuracy_func, i)
                      for i in range(self.k_validation)]
        if self.confusion:
            print(sum(accuracies))
            return 0, 0
        else:
            avg_accuracy = sum(accuracies) / len(accuracies)
            sd_accuracy = sqrt(sum(map(lambda acc: (acc - avg_accuracy) ** 2, accuracies)) / (self.k_validation - 1))
            return avg_accuracy, sd_accuracy

    @staticmethod
    def get_ith_accuracy(data_chunks, result_chunks, eta, accuracy_func, i):
        """
            Consumes the data, results, an accuracy function, and which iteration of the cross fold validation
            loop where in, and runs the test function with the training data as all of the data except for the
            ith chunk, which is reserved for the sample set
        :param data_chunks: a list of k sub-lists of the data, where k is the cross validation parameter
        :param result_chunks: a list of k sub-lists of the results, where k is the cross validation parameter
        :param eta: integer corresponding to the eta min percentage
        :param accuracy_func: a function which determines average accuracy k constructed tree with
            the given eta min using cross fold validation, where k is the k validation parameter
        :param i: which sub-list to omit from the training data
        :return: accuracy of the decision tree constructed from all but the ith sub-list and the given
            eta min parameter
        """
        mask_mat = [j != i for j in range(len(data_chunks))]
        mask = lambda lists: np.concatenate(list(compress(lists, mask_mat)))
        return accuracy_func(eta, DataStore.normalize_columns(mask(data_chunks)), mask(result_chunks),
                             DataStore.normalize_columns(np.array(data_chunks[i])), result_chunks[i])

    def accuracy(self, eta, data_chunks, result_chunks, test_data, test_results):
        """
            Uses our decision tree to test the given data
        :param eta: eta min for the tree to be built
        :param data_chunks: Training data
        :param result_chunks: Training data results
        :param test_data: test data
        :param test_results: results for test data
        :returns: The percentage of accurate predictions
        """
        d = self.tree_type(data=data_chunks, results=result_chunks, data_store=self, eta=eta)
        return d.test(test_data, test_results, with_confusion=self.confusion)

    def scikit_classifier(self, eta, data_chunks, result_chunks, test_data, test_results):
        """
            Uses scikit learns decision tree classifier to test the given data to compare our success
        :param eta: eta min for the tree to be built
        :param data_chunks: Training data
        :param result_chunks: Training data results
        :param test_data: test data
        :param test_results: results for test data
        :returns: The percentage of accurate predictions
        """
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion="entropy", min_samples_split=eta)
        clf.fit(DataStore.normalize_columns(data_chunks), result_chunks)
        return sum(map(eq, test_results, clf.predict(test_data))) / len(test_data)

    def scikit_regressor(self, eta, data_chunks, result_chunks, test_data, test_results):
        """
            Uses scikit learns decision tree classifier to test the given data to compare our success
        :param eta: eta min for the tree to be built
        :param data_chunks: Training data
        :param result_chunks: Training data results
        :param test_data: test data
        :param test_results: results for test data
        :param accuracies: list to add the accuracy of this test to
        """
        from sklearn.tree import DecisionTreeRegressor
        clf = DecisionTreeRegressor(min_samples_split=eta)
        clf.fit(data_chunks, result_chunks)
        predicted = clf.predict(test_data)
        return RegressionTree.mean_squared_error(test_results, predicted)


if __name__ == "__main__":
    ds = DataStore()
    print("Completed data load, beginning cross validation")
    ds.test(ds.start, ds.stop, ds.step, ds.test_func)