import argparse
import json
import numpy as np
from decision_tree import DecisionTree

__author__ = 'jamesmcnamara'


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
    parser.add_argument("-t", "--type", type=str,
                        help="Type of the columns in 'infile'. Options are: "
                             "'num', 'choice', 'str'. Must be homogeneous.")
    parser.add_argument("-m", "--meta", type=open, help="Meta file containing JSON formatted descriptions of the data")

    parser.add_argument("-c", "--cross", type=int, default=10,
                        help="Set the parameter for k-fold cross validation. Default 10")

    return parser.parse_args()


class DataStore:
    def __init__(self):
        # Load data
        parser = load_data()
        self.meta = json.load(parser.meta)
        self.width, self.height, self.type = self.meta["width"], self.meta["height"], self.meta["type"]
        self.data, self.results = self.extract(parser.infile, self.width, self.height)
        self.k_validation = parser.cross

        # Clean Data
        self.shuffle()
        self.normalize_columns()

        # Reference data
        self.result_types = set(self.results)
        self.result_map = {result: i for i, result in enumerate(self.result_types)}
        self.results = [self.result_map[val] for val in self.results]

        # Create the decision trees
        # start, stop, step = parser.range
        # self.d_trees = {eta_percent: DecisionTree(data_store=self, eta=int(len(self.data)/eta_percent))
        #                 for eta_percent in range(start, stop + 1, step)}

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
        return list(zip(*[iter(data)] * (len(data) // segments)))

    def find_best_tree(self, eta):
        data_chunks = self.chunk(self.data, self.k_validation)
        result_chunks = self.chunk(self.results, self.k_validation)

        for i in range(self.k_validation):
            test_data, test_results = data_chunks.pop(i), result_chunks.pop(i)
            d = DecisionTree(data=np.concatenate(data_chunks), results=np.concatenate(result_chunks),
                             data_store=self, eta=eta)
            accuracy = d.test(test_data, test_results)
            data_chunks.index(test_data, i)


ds = DataStore()

from code import interact
interact(local=locals())