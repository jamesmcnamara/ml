import argparse
import json
import numpy as np
from math import sqrt
from decision_tree import EntropyTree, RegressionTree, CategoricalEntropyTree

__author__ = 'jamesmcnamara'

tree_map = {"entropy": EntropyTree, "regression": RegressionTree, 'categorical': CategoricalEntropyTree}


def load_data():
    """
        Parses the command line arguments, and loads the arguments onto an ArgumentParser object, which it returns
    :return: an ArgumentParser object with all of the specified command line features written on
    """
    parser = argparse.ArgumentParser(description="Decision tree generator for CSV files")

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
    parser.add_argument("-d", "--debug", type=bool, default=False,
                        help="Use sci-kit learn instead of ours, to test that the behavior is correct")

    parser.add_argument("-cf", "--confusion", type=bool, default=False,
                        help="include a confusion matrix in the output")

    return parser.parse_args()


class DataStore:
    def __init__(self):
        # Load data
        parser = load_data()
        self.meta = json.load(parser.meta) if parser.meta \
            else json.load(open(parser.infile.name.replace(".csv", ".meta")))
        self.width, self.height, self.type = self.meta["width"], self.meta["height"], self.meta["type"]
        self.data, self.results = self.extract(parser.infile, self.width, self.height)
        self.k_validation = parser.cross
        self.tree_type = tree_map[parser.tree]
        self.confusion = parser.confusion
        self.debug = parser.debug

        # Clean Data
        self.shuffle()
        if self.tree_type != CategoricalEntropyTree:
            self.normalize_columns()

        # Reference data for classifiers
        if self.tree_type != RegressionTree:
            self.result_types = set(self.results)
            self.result_map = {result: i for i, result in enumerate(self.result_types)}
            self.results = [self.result_map[val] for val in self.results]
        else:
            self.results = list(map(float, self.results))

        print("Completed data load, beginning cross validation")
        # Create the decision trees
        start, stop, step = parser.range
        for eta_percent in range(start, stop + 1, step):
            if self.debug:
                test_func = self.scikit_classifier if self.tree_type == EntropyTree else self.scikit_regressor
            else:
                test_func = self.our_test

            avg, sd = self.cross_fold_validation(eta_percent, test_func)

            if self.tree_type != RegressionTree:
                print("{}% eta gave a classification accuracy of {:.2f}% with a standard deviation of {:.2f}%"
                      .format(eta_percent, avg * 100, sd * 100))
            else:
                print("{}% eta had an average MSE of {:.2f} with a standard deviation of {:.2f}%"
                      .format(eta_percent, avg, sd))

    def extract(self, file, width, height):
        """
            Extracts and loads the data from the given file into a numpy matrix and the
            results into a numpy array
        :param file: input CSV file with the first n-1 columns containing the observations and the
            last column containing the results from the training set
        :return: a tuple of the data in matrix form and an array of the result data
        """
        transform = lambda values: list(map(float if self.type == "float" else str, values))
        data = np.zeros((height, width - 1), self.type)
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

    def normalize_columns(self):
        """
            For all columns with numeric type, normalize their values to be between [0,1]
        """
        # Helper function which maps the normalization function over a column from the input matrix
        def column_mapper(col):
            minimum, maximum = min(col), max(col)
            return map(lambda x: (x - minimum) / (maximum - minimum), col)

        for index, norm_col in enumerate(column_mapper(col) for col in self.data.T):
            self.data[:, index] = list(norm_col)

    @staticmethod
    def chunk(data, segments):
        """
            Consumes a list and a number and returns a the input list split into n chunks in order
        """
        return list(zip(*[iter(data)] * (len(data) // segments)))

    def cross_fold_validation(self, eta, test_func):
        """
            Splits this data set into k chunks based on the k_validation attribute, and excludes each chunk,
            one at a time, from the input data set to a decision tree, and then tests that decision tree on
            the excluded data set. The function accumulates each of the accuracy measures, and returns the
            average accuracy and the standard deviation
        :param eta: eta min for the decision trees (early stopping strategy)
        :param test_func: a function to pass all of the relevant information too, which will append the accuracy
            of some test to our accuracies list
        :return: 2-tuple of average accuracy and standard deviation
        """

        data_chunks = self.chunk(self.data, self.k_validation)
        result_chunks = self.chunk(self.results, self.k_validation)
        accuracies = []
        for i in range(self.k_validation):
            test_data, test_results = data_chunks.pop(i), result_chunks.pop(i)
            test_func(len(self.data) * (eta / 100), np.concatenate(data_chunks), np.concatenate(result_chunks),
                      test_data, test_results, accuracies)
            data_chunks.insert(i, test_data)
            result_chunks.insert(i, test_results)

        if self.confusion:
            print(sum(accuracies))
            return 0, 0
        else:
            avg_accuracy = sum(accuracies) / len(accuracies)
            sd_accuracy = sqrt(sum(map(lambda acc: (acc - avg_accuracy) ** 2, accuracies)))
            return avg_accuracy, sd_accuracy

    def our_test(self, eta, data_chunks, result_chunks, test_data, test_results, accuracies):
        """
            Uses our decision tree to test the given data
        :param eta: eta min for the tree to be built
        :param data_chunks: Training data
        :param result_chunks: Training data results
        :param test_data: test data
        :param test_results: results for test data
        :param accuracies: list to add the accuracy of this test to
        :return:
        """
        d = self.tree_type(data=data_chunks, results=result_chunks, data_store=self, eta=eta)
        accuracies.append(d.test(test_data, test_results, with_confusion=self.confusion))

    def scikit_classifier(self, eta, data_chunks, result_chunks, test_data, test_results, accuracies):
        """
            Uses scikit learns decision tree classifier to test the given data to compare our success
        :param eta: eta min for the tree to be built
        :param data_chunks: Training data
        :param result_chunks: Training data results
        :param test_data: test data
        :param test_results: results for test data
        :param accuracies: list to add the accuracy of this test to
        :return:
        """
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion="entropy", min_samples_split=eta)
        clf.fit(np.concatenate(data_chunks), np.concatenate(result_chunks))
        accuracies.append(sum(map(lambda arr_and_result: arr_and_result[0] == arr_and_result[1],
                                  zip(clf.predict(test_data), test_results))) / len(test_data))

    def scikit_regressor(self, eta, data_chunks, result_chunks, test_data, test_results, accuracies):
        """
            Uses scikit learns decision tree classifier to test the given data to compare our success
        :param eta: eta min for the tree to be built
        :param data_chunks: Training data
        :param result_chunks: Training data results
        :param test_data: test data
        :param test_results: results for test data
        :param accuracies: list to add the accuracy of this test to
        :return:
        """
        from sklearn.tree import DecisionTreeRegressor
        print("Eta is {}".format(eta))
        clf = DecisionTreeRegressor(min_samples_split=eta)
        clf.fit(np.concatenate(data_chunks), np.concatenate(result_chunks))
        predicted = clf.predict(test_data)
        accuracies.append(RegressionTree.mean_squared_error(test_results, predicted))

ds = DataStore()