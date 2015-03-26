from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import compress
from math import sqrt
from operator import eq
from os.path import basename
from re import match

import argparse
import json
import numpy as np
from statistics import stdev, mean

from dtree.classification_tree import EntropyTree, CategoricalEntropyTree
from dtree.regression_tree import RegressionTree
from text_classification.NaiveBayes import NaiveBayes
from regression.regression import Regressor
from utils.normalization import normalize, normalize_validation

__author__ = 'jamesmcnamara'


def rmse(predictions, actual):
    return sqrt(sum((prediction - observed) ** 2 for prediction, observed in zip(predictions, actual)) / len(actual))


def load_args():
    """
        Parses the command line arguments, and loads the arguments onto an ArgumentParser object, which it returns
    :return: an ArgumentParser object with all of the specified command line features written on
    """
    parser = argparse.ArgumentParser(description="General purpose command line machine learning tool.")

    parser.add_argument("infile", type=open, help="CSV file with training data")

    parser.add_argument("-n", "--normalization", type=str, default="arithmetic",
                        help="What type of normalization to apply to the data. Options are "
                             "arithmetic or z-score. Default 'arithmetic'")

    parser.add_argument("-m", "--meta", type=open, default=None,
                        help="Meta file containing JSON formatted descriptions of the data")

    parser.add_argument("-gm", "--generate-meta", action="store_true", default=False,
                        help="Meta file containing JSON formatted descriptions of the data")

    parser.add_argument("-d", "--debug", action="store_true", default=False,
                        help="Use sci-kit learn instead of learn.py, to test that the behavior is correct")

    parser.add_argument("-cv", "--cross", type=int, default=10,
                        help="Set the parameter for k-fold cross validation. Default 10")

    parser.add_argument("-v", "--validation", type=open,
                        help="Separate validation set to use instead of cross-fold")

    parser.add_argument("-i", "--interactive", action="store_true",
                        help="After loading the data, drop the user into an interactive REPL")

    # Decision Tree options
    parser.add_argument("-r", "--range", type=int, nargs=3,
                        help="Range of eta values to use for cross validation. The first "
                             "value is start, the second is end, and the last is interval")

    parser.add_argument("-t", "--tree", type=str, default="entropy",
                        help="What type of decision tree to build for the data. Options are "
                             "'entropy', 'regression', or 'categorical'. Default 'entropy'")

    parser.add_argument("-cf", "--with-confusion", action="store_true", default=False,
                        help="Include a confusion matrix in the output")

    parser.add_argument("-b", "--binary-splits", action="store_true", default=False,
                        help="Convert a multi-way categorical matrix to a binary matrix")

    # Regression options
    parser.add_argument("-reg", "--regression", action="store_true", default=False,
                        help="Develop a linear model using regression for the given data set")

    parser.add_argument("-pw", "--powers", type=int, default=1,
                        help="Pad out the data set with powers of the input sets")

    parser.add_argument("-gd", "--gradient-descent", action="store_true", default=False,
                        help="Use gradient descent to regress the given matrix")

    parser.add_argument("-pr", "--plot-rmse", action="store_true", default=False,
                        help="Display a graph showing the iterative root mean squared error "
                             "of each iteration of the gradient descent")

    parser.add_argument("-pp", "--plot-powers", action="store_true", default=False,
                        help="Display a graph showing the iterative root mean squared error "
                             "of each iteration of the gradient descent")

    parser.add_argument("-s", "--step", type=float, default=0,
                        help="Develop a linear model using regression for the given data set")

    parser.add_argument("-tr", "--tolerance", type=float, default=0,
                        help="Epsilon value to use during gradient descent")

    parser.add_argument("-it", "--iterations", type=int, default=1000,
                        help="Maximum number of iterations for gradient descent")

    #Text classification options
    parser.add_argument("-tx", "--text", action="store_true", default=False,
                        help="Use a text classification algorithm")

    parser.add_argument("-ev", "--event-model", type=str, default="multinomial",
                        help="Which event model to use. Choices are 'bern' for multivariate bernoulli "
                             "or 'multinomial' for multinomial. Default multinomial.")


    parser.add_argument("-vs", "--vocab-size", type=int, default=100,
                        help="The number of most frequently occurring words to keep. Negatives cause flips")


    parser.add_argument("-tsp", "--test-prefix", type=str, help="File prefix")
    parser.add_argument("-trp", "--train-prefix", type=str, help="File prefix")

    return parser.parse_args()


def generate_meta(file):
    """
        Generates the meta data file for a given csv that is used for initializing the data structures
    :param file: data set for which to generate a meta file
    """
    is_float = lambda val: len(match(r"[-+]?\d+\.?\d*", val).group(0)) == len(val) if match(r"[-+]?\d+\.?\d*", val) else False
    float_data = float_result = 0
    for count, line in enumerate(file):
        *data, result = line.split(",")
        float_data += all(map(is_float, data))
        float_result += is_float(result.strip("\n"))

    with open(file.name.replace(".csv", ".meta"), "w+") as f:
        f.write(json.dumps(dict(name=basename(file.name).replace(".csv", ""), width=len(line.split(",")),
                                height=count + 1, data_type="float" if float_data > (count / 2) else "str",
                                result_type="float" if float_result > (count / 2) else "str"), sort_keys=True,
                           indent=4, separators=(',', ': ')))


class DataStore:
    __metaclass__ = ABCMeta

    def __init__(self, parser):
        # Load data
        self.meta = json.load(parser.meta) if parser.meta \
            else json.load(open(parser.infile.name.replace(".csv", ".meta")))

        self.width, self.height, self.data_type, self.result_type = \
            self.meta["width"], self.meta["height"], self.meta["data_type"], self.meta["result_type"]

        # ETL Data
        self.data, self.results = self.extract(self.data_type, self.result_type, self.width, self.height, parser.infile)
        if parser.validation:
            validation_meta = json.load(open(parser.validation.name.replace("Train", "Validation").replace(".csv", ".meta")))
            self.validation_data, self.validation_results = \
                self.extract(validation_meta["data_type"], validation_meta["result_type"],
                             validation_meta["width"], validation_meta["height"], parser.validation)

        if parser.powers > 1:
            self.data = self.add_powers(self.data, parser.powers)
            if parser.validation:
                self.validation_data = self.add_powers(self.validation_data, parser.powers)

        # Number of folds to perform cross validation on
        self.k_validation = parser.cross if parser.cross != 1 else len(self.data)

        # Whether or not to use built in libraries to test the algorithms
        self.debug = parser.debug

        # Normalization method to use
        self.normalization = parser.normalization

        # Number of powers to pad out the dataset
        self.powers = parser.powers

        # Clean Data
        self.shuffle()

    @staticmethod
    def extract(data_type, result_type, width, height, file):
        """
            Extracts and loads the data from the given file into a numpy matrix and the
            results into a numpy array
        :param file: input CSV file with the first n-1 columns containing the observations and the
            last column containing the results from the training set
        :return: a tuple of the data in matrix form and an array of the result data
        """
        transform = lambda values: list(map(float if data_type == "float" else str, values))
        result_trans = lambda result: float(y) if result_type == "float" else y
        data = np.zeros((height, width - 1), data_type)
        results = [""] * height
        for i, line in enumerate(file):
            *xs, y = line.split(",")
            data[i] = transform(xs)
            results[i] = result_trans(y.strip("\n"))
        return data, results

    @staticmethod
    def add_powers(data, power):
        """
            Consumes a data set, and adds columns for every power up to power
        :param data: input design matrix
        :param power: number of sets of columns to add of higher powers of the input columns
        :return: data with the power columns added
        """
        exp = lambda p: lambda x: x ** p
        h, w = data.shape
        mat = np.zeros((h, w * power), data.dtype)
        for i, input_column in enumerate(data.T):
            for j in range(power):
                jth_power = exp(j + 1)
                mat.T[w * j + i] = list(map(jth_power, input_column))
        return mat

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


class CrossFoldMixin:

    @staticmethod
    def cross_validation(accuracy_func, data, results, k_validation, normalization, *args, **kwargs):
        """
            Splits this data set into k chunks based on the k_validation attribute, and excludes each chunk,
            one at a time, from the input data set to a decision tree, and then tests that decision tree on
            the excluded data set. The function accumulates each of the accuracy measures, and returns the
            average accuracy and the standard deviation
        :param eta: eta min for the decision trees (early stopping strategy)
        :param accuracy_func: a function which determines average accuracy k constructed tree with
            the given eta min using cross fold validation, where k is the k validation parameter
        :return: A list of accuracies across the folds
        """

        data_chunks = CrossFoldMixin.chunk(data, k_validation)
        result_chunks = CrossFoldMixin.chunk(results, k_validation)
        return [CrossFoldMixin.get_ith_accuracy(data_chunks, result_chunks, accuracy_func, i, normalization, *args, **kwargs)
                for i in range(k_validation)]

    @staticmethod
    def get_ith_accuracy(data_chunks, result_chunks, accuracy_func, i, normalization, *args, **kwargs):
        """
            Consumes the data, results, an accuracy function, and which iteration of the cross fold validation
            loop where in, and runs the test function with the training data as all of the data except for the
            ith chunk, which is reserved for the sample set
        :param data_chunks: a list of k sub-lists of the data, where k is the cross validation parameter
        :param result_chunks: a list of k sub-lists of the results, where k is the cross validation parameter
        :param accuracy_func: a function which determines average accuracy k constructed tree with
            the given eta min using cross fold validation, where k is the k validation parameter
        :param i: which sub-list to omit from the training data
        :return: accuracy of the decision tree constructed from all but the ith sub-list and the given
            eta min parameter
        """
        mask_mat = [j != i for j in range(len(data_chunks))]
        mask = lambda lists: np.concatenate(list(compress(lists, mask_mat)))
        return accuracy_func(mask(data_chunks), mask(result_chunks), np.array(data_chunks[i]), np.array(result_chunks[i]), normalization, *args, **kwargs)

    @staticmethod
    def chunk(data, segments):
        """
            Consumes a list and a number and returns a the input list split into n chunks in order
        """
        return list(zip(*[iter(data)] * (len(data) // segments)))


class DecisionTreeDataStore(DataStore, CrossFoldMixin):

    tree_map = {"regression": RegressionTree, "entropy": EntropyTree, "categorical": CategoricalEntropyTree}

    def __init__(self, parser):
        super().__init__(parser)
        self.tree_type = DecisionTreeDataStore.tree_map[parser.tree]
        self.confusion = parser.with_confusion
        self.start, self.stop, self.step = parser.range

        # Clean Data
        self.shuffle()
        if parser.binary_splits:
            self.data = self.binarize_columns(self.data)

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

    def test(self):
        """
            Consumes descriptors of various eta levels and an accuracy function and determines
            prints messages about the average accuracy and variation across the folds
        :param start: first eta min value to test
        :param stop: last eta min value to test
        :param step: strides for updating eta mins from start to stop
        :param accuracy_func: a function which determines average accuracy k constructed tree with
            the given eta min using cross fold validation, where k is the k validation parameter
        """
        for eta_percent in range(self.start, self.stop + 1, self.step):
            accuracies = self.cross_validation(self.accuracy, self.data, self.results,
                                               self.k_validation, self.normalization,  eta_percent)
            if self.confusion:
                print(sum(accuracies))
            else:
                avg = mean(accuracies)
                sd = stdev(accuracies)

                if self.tree_type != RegressionTree:
                    print("{}% eta gave a classification accuracy of {:.2f}% with a standard deviation of {:.2f}%"
                          .format(eta_percent, avg * 100, sd * 100))
                else:
                    print("{}% eta had an average MSE of {:.2f} with a standard deviation of {:.2f}%"
                          .format(eta_percent, avg, sd))

    def accuracy(self, data_chunks, result_chunks, test_data, test_results, normalization, eta, *args):
        """
            Uses our decision tree to test the given data
        :param eta: eta min for the tree to be built
        :param data_chunks: Training data
        :param result_chunks: Training data results
        :param test_data: test data
        :param test_results: results for test data
        :returns: The percentage of accurate predictions
        """
        d = self.tree_type(data=normalize(normalization, data_chunks), results=result_chunks, data_store=self, eta=eta)

        return sum(map(lambda exp, act: exp == act,
                       d.predict(normalize_validation(normalization, data_chunks, test_data)),
                       test_results)) / len(test_results)

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


    def scikit_classifier(self, eta, data_chunks, result_chunks, test_data, test_results, normalization):
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
        clf.fit(normalize(normalization, data_chunks), result_chunks)
        return sum(map(eq, test_results, clf.predict(test_data))) / len(test_data)

    def scikit_regressor(self, eta, data_chunks, result_chunks, test_data, test_results, normalization):
        """
            Uses scikit learns decision tree classifier to test the given data to compare our success
        :param eta: eta min for the tree to be built
        :param data_chunks: Training data
        :param result_chunks: Training data results
        :param test_data: test data
        :param test_results: results for test data
        :param normalization: type of normailization to use
        """
        from sklearn.tree import DecisionTreeRegressor
        clf = DecisionTreeRegressor(min_samples_split=eta)
        clf.fit(data_chunks, result_chunks)
        predicted = clf.predict(test_data)
        return RegressionTree.mean_squared_error(test_results, predicted)


class RegressionDataStore(DataStore, CrossFoldMixin):

    def __init__(self, parser):
        super().__init__(parser)
        self.use_gradient = parser.gradient_descent
        self.collect_rmse = parser.plot_rmse
        if self.use_gradient:
            self.tolerance = parser.tolerance
            self.step = parser.step
            self.iterations = parser.iterations

        self.test_func = self.accuracy

    def test(self, accuracy_func, *args, **kwargs):
        if hasattr(self, "validation_data"):
            return self.accuracy(self.data, self.results, self.validation_data, self.validation_results, **kwargs), 0
        accuracies = self.cross_validation(accuracy_func, self.data, self.results, self.k_validation, self.normalization, **kwargs)
        avg = mean(accuracies)
        sd = stdev(accuracies)
        return avg, sd

    def accuracy(self, training_data, training_results, test_data, test_results, normalization, **kwargs):
        """
            Uses our decision tree to test the given data
        :param data_chunks: Training data
        :param result_chunks: Training data results
        :param test_data: test data
        :param test_results: results for test data
        :returns: The percentage of accurate predictions
        """
        r = Regressor(training_data, training_results, self, **kwargs)
        return rmse(r.predict(test_data), test_results)


class TextDataStore:

    def __init__(self, parser):
        self.model = parser.event_model
        self.size = parser.vocab_size + 1
        data, self.vocab = self.extract(parser.train_prefix, self.size, parser.event_model)
        self.data = self.restrict_data(data, self.vocab)

        test, self.result_vocab = self.extract(parser.test_prefix, self.size, parser.event_model)
        self.test = self.restrict_data(test, self.vocab)

        get_labels = lambda prefix: list(map(lambda line: int(line.split()[0]), open(prefix + ".label")))
        self.data_labels = get_labels(parser.train_prefix)
        self.result_labels = get_labels(parser.test_prefix)

        self.breakpoints = self.get_label_breakpoints(self.data_labels)

    @staticmethod
    def extract(prefix, head, event_model):
        data_file = open(prefix + ".data")
        meta = json.load(open(prefix + ".meta"))
        word_count = meta["word_count"] + 1
        doc_count = meta["docs"]

        data = np.zeros((doc_count, word_count), dtype=np.int)
        freq = np.zeros((word_count, 2), dtype=np.int)

        for line in data_file:
            doc_id, word_id, count = map(int, line.split())
            data[doc_id - 1, word_id] = count if event_model == "multinomial" else 1

            freq[word_id, 0] = word_id
            freq[word_id, 1] += count

        return data, freq[freq[:, -1].argsort()][:-head:-1, 0]

    @staticmethod
    def restrict_data(data, indicies):
        trimmed_data = np.zeros((len(data), len(indicies)), np.int32)
        for i in range(len(data)):
            trimmed_data[i] = [data[i, j] for j in indicies]
        return trimmed_data

    @staticmethod
    def get_label_breakpoints(labels):
        current_label = 0
        breakpoints = []
        for i, label in enumerate(labels):
            if label != current_label:
                breakpoints.append(i)
                current_label = label
        breakpoints.append(i)
        return breakpoints

if __name__ == "__main__":
    parser = load_args()
    if parser.generate_meta:
        generate_meta(parser.infile)
    if parser.text:
        ts = TextDataStore(parser)
        clf = NaiveBayes(ts)
        predictions = clf.predict(ts.test)
        from code import interact
        interact(local=locals())
    elif parser.regression:
        ds = RegressionDataStore(parser)
        if parser.interactive:
            from code import interact
            interact(local=locals())

        print("Average was: {}, Standard deviation was {}".format(*ds.test(ds.test_func)))
    else:
        ds = DecisionTreeDataStore(parser)
        ds.test()