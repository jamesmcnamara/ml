import numpy as np
from math import log2
from abc import ABCMeta, abstractmethod


def lg(num):
    """
        Binary log of input, with the exception that lg(0) is 0
    :param num: number to lg
    :return: base 2 log of num
    """
    return log2(num) if num else 0


class DecisionTree:

    __metaclass__ = ABCMeta

    def __init__(self, data_store=None, eta=0, data=None, results=None, parent=None):
        if parent:
            assert data is not None and results is not None, \
                "You must pass in all three of data, results and parent or none at all " \
                "to the constructor of the decision tree"
            self.parent = parent
            self.data_store = parent.data_store
            self.eta = self.parent.eta
            self.used_columns = list(parent.used_columns)
        else:
            assert data_store is not None and eta is not 0 and data is not None and results is not None, \
                "When creating a DecisionTree without a parent, you must supply a DataStore, data, results and eta min"
            self.data_store = data_store
            self.eta = eta
            self.used_columns = []
            self.parent = None

        self.data, self.results = data, results
        self.split_on, self.splitter = -1, None
        self.left, self.right = None, None

        # print("Data length: {}, eta min: {}, split: {}".format(len(self.data), self.eta, len(self.data) > self.eta))l
        # If the number of nodes in this tree is above the eta min, the measure function is non-zero,
        # and not all columns have been used, split this tree
        if len(self.data) >= self.eta and self.measure_function(self.results) \
                and len(self.used_columns) < len(self.data.T):
            self.split()
            self.classification = None
        else:
            self.classification = self.value()

    @abstractmethod
    def value(self):
        """
            Determines what value nodes in this node should be applied
        :return: the value for nodes in this tree
        """
        raise NotImplementedError("Concrete subclasses of decision tree must "
                                  "implement their own leaf classification method")

    @abstractmethod
    def test(self, data, results, with_confusion=False):
        """
            Returns a measure of the accuracy of classifying the observation in data using this tree
        :param data: observations that this tree was not trained on
        :param results: the related results
        :return: Some measure of the accuracy
        """
        raise NotImplementedError("Concrete subclasses of decision tree must "
                                  "implement their own testing method")


    def split(self):
        """
            Splits this DecisionTree on the column that provides the most information gain
            if there is more than eta nodes in this tree
        """
        self.split_on, self.splitter = self.best_attr(self.data, self.results)
        if not self.splitter:
            self.classification = self.value()
        else:
            # Save bookkeeping information about which column was split on
            self.used_columns.append(self.split_on)

            left, left_results, right, right_results = [], [], [], []

            # Create right and left data sets by using the splitter to split on column
            for i, (row, result) in enumerate(zip(self.data, self.results)):
                if self.splitter(row[self.split_on]):
                    left.append(row)
                    left_results.append(result)
                else:
                    right.append(row)
                    right_results.append(result)
            left, right = np.array(left), np.array(right)

            # Create right and left children of the same type as this class
            self.left = self.__class__(data=left, results=left_results, parent=self)
            self.right = self.__class__(data=right, results=right_results, parent=self)

    def best_attr(self, data, results):
        """
            Determines the best column to split the input matrix on, and returns the column index and splitter function
        :param data: input matrix
        :param results: the classifications that correspond to the rows of the input matrix data
        :return: the a 2-tuple of column-index and a splitter function
        """
        best_gain, best_column, best_splitter = -1, -1, None
        for i, column in enumerate(data.T):
            if i not in self.used_columns:
                gain, splitter = self.best_binary_split(column, results)
                if gain > best_gain:
                    best_gain, best_column, best_splitter = gain, i, splitter
        return best_column, best_splitter

    def best_binary_split(self, attr, results):
        """
            Iterates over every interval of attr, calculating the information gain based on partitioning at that
            interval, eventually returning the maximum information gain obtainable by partitioning on attr, and the
            corresponding splitter
        :param attr: The column of attribute data to split results on
        :param results: The corresponding results to be split
        :return: tuple of float measuring maximum information gain obtainable by partitioning on attr, and the
            corresponding splitter function
        """
        splitter = lambda interval: lambda x: x <= interval
        gain = {self.gain(attr, results, splitter(interval)): splitter(interval)
                for interval in sorted(set(attr))[:-1]}
        return (max(gain), gain[max(gain)]) if gain else (-1, None)

    def gain(self, attr, results, bin_splitter):
        """
            Measures the expected gain as measured by the trees gain_function by splitting the results set on attr
            via bin_splitter
        :param attr: 1D array of attr values corresponding to the entries in results
        :param results: 1D array of classifications, with i-th classification having attribute attr[i]
        :param bin_splitter: a function that consumes an instance of the attr array and outputs a boolean indicating
            which partition to categorize the classification
        :return: Float corresponding to the gain derived from partitioning on attr with bin_splitter
        """
        add_elements_if = lambda val: [result for element, result in zip(attr, results) if bin_splitter(element) == val]

        left, right = add_elements_if(True), add_elements_if(False)
        if len(left) == 0 or len(right) == 0:
            import code
            code.interact(local=locals())
        return self.measure_function(results) - sum(len(partition) / len(results) * self.measure_function(partition)
                                                    for partition in (left, right))

    @abstractmethod
    def measure_function(self, results):
        """
            ABSTRACT: consumes a results array and outputs some custom measure functions value
        :param results: The results for this node
        :return: Float corresponding to the measure derived from the results data
        """
        raise NotImplementedError("The DecisionTree is only a scaffolding class. It must be "
                                  "sub-classed and the 'measure_function' method must be "
                                  "implemented. See EntropyTree, and RegressionTree for examples")

    def depth(self, initial_count=0):
        """
            Determine how deep in the tree we are
        """
        return self.parent.depth(initial_count=initial_count + 1) if self.parent else initial_count

    def node_counts(self, acc):
        """
            Return a list of
        :param acc:
        :return:
        """
        if self.left:
            self.left.node_counts(acc)
        if self.right:
            self.right.node_counts(acc)
        if self.left is None and self.right is None:
            acc.append(len(self.data))
        return acc

    def __len__(self):
        """
            OVERRIDE: len(self) returns the number of entries in this node
        """
        return len(self.data)

    def __repr__(self):
        """
            OVERRIDE: repr(self) returns a pretty printed version of the tree
        """
        return str(self)

    def __getitem__(self, item):
        """
            OVERRIDE: supports indexing by arbitrary strings of 'l' and 'r' to return a corresponding subtree
            evaluated from left-to-right.
            e.g. self["llr"] returns the self.left.left.right
        """
        assert item.count("l") + item.count("r") == len(item), "Indexing must include only 'l' and 'r' characters"
        if len(item) == 1:
            return self.left if item == "l" else self.right

        char, *rest = item
        if char == 'l':
            return self.left["".join(rest)]
        else:
            return self.right["".join(rest)]


class EntropyTree(DecisionTree):
    def entropy(self, results):
        """
            Calculates the total dispersion in the input classifications by measure of the asymptotic
            bit rate transfer requirements
        :param results: A 1D array of classifications
        :return: float representing entropy of input set
        """
        result_dist = [0] * len(self.data_store.result_types)
        element_count = len(results)

        for result in results:
            result_dist[result] += 1

        return -sum(lg(count / element_count) * (count / element_count) for count in result_dist if count)

    def measure_function(self, results):
        return self.entropy(results)

    def value(self):
        """
            Classifies nodes in this tree by which result is most frequent in this node
        :return:
        """
        most_common_class, max_count = 0, 0
        for i in range(len(self.data_store.result_types)):
            if self.results.count(i) > max_count:
                most_common_class, max_count = i, self.results.count(i)
        return most_common_class

    def test(self, data, results, with_confusion=False):
        """
            Consumes test observations and their results and returns the percentage of entries that this tree
            classified correctly
        :param data: matrix of observational data that the tree was not trained on
        :param results: array of resulting data, with the indices matching the rows of data
        :return: percent of entries that were correctly classified
        """
        if with_confusion:
            confusion = np.zeros((len(self.data_store.result_types), len(self.data_store.result_types)))
            for row, result in zip(data, results):
                classification = self.classify(row)
                confusion[result, classification] += 1
            return confusion
        else:
            return sum(map(self.match, zip(data, results))) / len(data)

    def classify(self, row):
        """
            Consumes an observation returns the classification of that observation via this tree and if this tree is a leaf,
            determines 1 if this observation was classified correctly else 0
            else uses this trees splitter to determine which sub-tree to delegate to, and then returns whether
            the subtree correctly classified the node
        :param row_and_result: a 2-tuple of observational data (1D array) and the result for that observation
        :return: 1 if this tree correctly classified the input else 0
        """
        if self.splitter:
            if self.splitter(row[self.split_on]):
                return self.left.classify(row)
            else:
                return self.right.classify(row)
        else:
            return self.classification

    def match(self, row_and_result):
        """
            Determines if the given observation was classified correctly by this tree
        :param row_and_result: a 2-tuple of observational data (1D array) and the result for that observation
        :return: True if this tree correctly classified the input else False
        """
        row, result = row_and_result
        return self.classify(row) == result

    def __str__(self):
        """
            OVERRIDE: str(self) returns a pretty printed version of the tree
        """
        offset = "\n" + "\t" * self.depth()
        s = "{off}Total Entries: {len},{off}Entropy: {ent:.3f}"\
            .format(off=offset, len=len(self), ent=self.entropy(self.results), col=self.split_on)
        s += ",{off}Split on column: {col}".format(off=offset, col=self.split_on) if self.split_on != -1 else ""

        for branch, branch_name in [(self.left, "Left"), (self.right, "Right")]:
            if branch:
                s += "{off}__________________________{off}{name}:{branch}"\
                    .format(off=offset, name=branch_name, branch=str(branch))
        return s


class RegressionTree(DecisionTree):

    # Set the measure function be the mean squared error of the given set against the average of that set
    measure_function = lambda _, results: \
        RegressionTree.mean_squared_error(results, RegressionTree.avg_generator(results))

    @staticmethod
    def mean_squared_error(actual, predicted):
        """
            Consumes a results set and a predicted set of equal length and outputs the mean squared error
        :param actual: The real observed results for this node (must be a list)
        :param predicted: The real observed results for this node (must be a list)
        :return: Mean squared error if all nodes in this set are predicted using the midpoint value
        """
        return sum((y - yhat)**2 for y, yhat in zip(actual, predicted)) / len(actual)

    @staticmethod
    def residual_sum_of_squares(results):
        """
            Calculates the sum of the squared deviations about the mean of a data set
        :param results: The real observed results for this node (must be a list)
        :return: the sum of the squared deviations about the mean of a data set
        """
        avg = sum(results) / len(results)
        return sum((y - avg)**2 for y in results)

    @staticmethod
    def r2(actual, predicted):
        """
            Measures the coefficient of determination of the regression tree, by 1 - u/v, where u is the sum of
            squared errors and and v is the residual sum of squares
        :param actual: the actual results from observations
        :param predicted: the predicted results via the regression tree
        :return: a value from 0-1 measuring the closeness of the model
        """
        return 1 - ((RegressionTree.mean_squared_error(actual, predicted) * len(actual)) /
                    RegressionTree.residual_sum_of_squares(actual))

    @staticmethod
    def avg_generator(results):
        """
            Consumes a list of numbers and return a generator which yields the average forever
        :param results: 1D array of integers
        :return: iterator which always yields the average of the input set
        """
        avg = sum(results) / len(results)
        while True:
            yield avg

    def value(self):
        """
            Classifies nodes in this tree by the average of the results data set
        :return:
        """
        return sum(self.results) / len(self.results)

    def test(self, data, results, with_confusion=False):
        """
            Consumes test observations and their results and returns the mean
            squared error obtained by classifying the data with this tree
        :param data: matrix of observational data
        :param results: array of resulting data, with the indices matching the rows of data
        :return: mean squared error obtained by classifying the data with this tree
        """
        return self.mean_squared_error(results, (self.regress(row, result) for row, result in zip(data, results)))

    def regress(self, row, result):
        """
            Returns the error obtained by
        :param row:
        :param result:
        :return:
        """
        if self.splitter:
            if self.splitter(row[self.split_on]):
                return self.left.regress(row, result)
            else:
                return self.right.regress(row, result)
        else:
            return self.value()