import numpy as np
from math import log2
from abc import ABCMeta, abstractmethod
from collections import namedtuple, defaultdict, Counter

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
        exists = lambda *l: all(map(lambda element: element is not None and element is not 0, l))

        assert exists(data, results, parent) or exists(data_store, results, data, eta), \
            "To construct a DecisionTree, you must either pass in keyword arguments " \
            "for data and results and either parent, or eta and data_store"
        self.parent = parent
        self.data_store = data_store or parent.data_store
        self.data, self.results = data, results
        self.eta = len(self.data) * (eta / 100) if eta else self.parent.eta
        self.used_columns = list(parent.used_columns) if parent else []

        # Stores which column this tree was split on, and possibly a function that consumes a value
        self.ColumnInfo = namedtuple("ColumnInfo", ["column", "gain", "splitter"])
        self.split_on = self.ColumnInfo(-1, -1, None)

        # If the number of nodes in this tree is above the eta min, the measure function is non-zero,
        # and not all columns have been used, split this tree
        if len(self.data) >= self.eta and self.measure_function(self.results) \
                and len(self.used_columns) < len(self.data.T):
            self.split()
            self.classification = None
        else:
            self.classification = self.value()

    def split(self):
        """
            Splits this DecisionTree on the column that provides the
            most gain if there is more than eta nodes in this tree
        """
        self.split_on = self.best_attr(self.data, self.results)
        if self.split_on.column == -1:
            self.classification = self.value()
        else:
            # Save bookkeeping information about which column was split on
            self.used_columns.append(self.split_on.column)
            self.partition(**self.split_on._asdict())

    def best_attr(self, data, results):
        """
            Determines the best column to split the input matrix on, and returns the column index and splitter function
        :param data: input matrix
        :param results: the classifications that correspond to the rows of the input matrix data
        :return: a ColumnInfo tuple
        """

        col_map = {i: self.best_split(i, column, results)
                   for i, column in enumerate(data.T)
                   if i not in self.used_columns}

        return max(col_map.values(), key=lambda col_info: col_info.gain)

    def measure_gain(self, results, *partitions):
        """
            Determines how much would be gained on the measure function by splitting results into partitions
        :param results: a 1D dimension of results
        :param partitions: an arbitrary number of partitions that the results are being split into
        :return: total reduction in the measure function
        """
        return self.measure_function(results) - sum(len(partition) / len(results) * self.measure_function(partition)
                                                    for partition in partitions)

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
    @abstractmethod
    def partition(self, **kwargs):
        """
            Partitions this tree based on the named tuple arguments handed in through kwargs
            possible parameters include:
            column: column to partition on
            splitter: function to use to split the given column
            gain: total gain passed in

        """
        raise NotImplementedError("Concrete subclasses of decision tree must "
                                  "implement their own best_split method")

    @abstractmethod
    def best_split(self, col_index, column, results):
        """
            Consumes a column and associated data, and returns the best gain achievable from splitting on that column
            and optionally a splitter function that can be used to split the data
        :param col_index: the index of the given column
        :param column: 1D array of observations
        :param results: corresponding results
        :return: ColumnInfo tuple holding the best gain achievable on this column (float), this column index,
         and possibly a splitter for it (function)
        """
        raise NotImplementedError("Concrete subclasses of decision tree must "
                                  "implement their own best_split method")


    @abstractmethod
    def value(self):
        """
            Determines what value nodes in this node should be applied
        :return: the value for nodes in this tree
        """
        raise NotImplementedError("Concrete subclasses of decision tree must "
                                  "implement their own leaf classification method")


    @abstractmethod
    def predict(self, data):
        """
            Returns a stream of classifications of the given observations in data using this tree
        :param data: observations that this tree was not trained on
        :return: stream of classifications
        """
        raise NotImplementedError("Concrete subclasses of decision tree must "
                                  "implement their own testing method")

    def depth(self, initial_count=0):
        """
            Determine how deep in the tree we are
        """
        return self.parent.depth(initial_count=initial_count + 1) if self.parent else initial_count

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


class BinaryTree(DecisionTree):

    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.left, self.right = None, None
        super().__init__(**kwargs)

    def partition(self, column=-1, splitter=None, **kwargs):
        """
            Partitions this data set into a left and right section based on the given column index
        :param column: index of the column to split on
        :param splitter: a function that consumes a value of the column, and returns whether it goes
            into the left or right sub tree
        """
        left, left_results, right, right_results = [], [], [], []
        if splitter is None:
            import code
            code.interact(local=locals())
        # Create right and left data sets by using the splitter to split on column
        for i, (row, result) in enumerate(zip(self.data, self.results)):
            if splitter(row[column]):
                left.append(row)
                left_results.append(result)
            else:
                right.append(row)
                right_results.append(result)
        left, right = np.array(left), np.array(right)

        # Create right and left children of the same type as this class
        self.left = self.__class__(data=left, results=left_results, parent=self)
        self.right = self.__class__(data=right, results=right_results, parent=self)

    def best_split(self, col_index, column, results):
        """
            Iterates over every interval of attr, calculating the information gain based on partitioning at that
            interval, eventually returning the maximum information gain obtainable by partitioning on attr, and the
            corresponding splitter
        :param column: The column of attribute data to split results on
        :param results: The corresponding results to be split
        :return: namedtuple of float measuring maximum information gain obtainable by partitioning on attr, and the
            corresponding splitter function
        """
        splitter = lambda interval: lambda x: x <= interval
        gain = {self.gain(column, results, splitter(interval)): splitter(interval)
                for interval in sorted(set(column))[:-1]}
        return self.ColumnInfo(column=col_index, gain=max(gain), splitter=gain[max(gain)]) \
            if gain else self.ColumnInfo(column=col_index, gain=-1, splitter=None)

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
        return self.measure_gain(results, left, right)

    def node_counts(self, acc):
        """
            Return a list of the number of observations in each leaf
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


class CategoricalTree(DecisionTree):

    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.children = {}
        super().__init__(**kwargs)

    def partition(self, column=-1, **kwargs):
        """
            Partitions this tree on the given column
        :param column: index of a column
        """
        symbol_to_data = defaultdict(lambda: ([], []))
        for row, result in zip(self.data, self.results):

            # Look up the symbol in the current row in the children dictionary, and adds the
            # current row and result to the collections stored at that symbol
            data, results = symbol_to_data[row[column]]
            data.append(row)
            results.append(result)

        self.children = {symbol: self.__class__(data=np.array(data), results=results, parent=self)
                         for symbol, (data, results) in symbol_to_data.items()}

    def best_split(self, col_index, column, results):
        """
            Consumes a column and associated data, and returns the best gain achievable from splitting on that column
            and optionally a splitter function that can be used to split the data
        :param column: 1D array of observations
        :param results: corresponding results
        :return: best gain achievable on that column (float) and possibly a splitter for it (function)
        """
        partitions = defaultdict(list)
        for value, result in zip(column, results):
            partitions[value].append(result)

        return self.ColumnInfo(column=col_index, gain=self.measure_gain(results, *partitions.values()), splitter=None)


class EntropyMixin(DecisionTree):

    __metaclass__ = ABCMeta

    def measure_function(self, results):
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

    def value(self):
        """
            Determines what value observations in this node should be applied
        :return: the value for nodes in this tree
        """
        [(most_common_class, _)] = Counter(self.results).most_common(1)
        return most_common_class

    def predict(self, data):
        """
            Consumes test observations classifies them based on the heuristics of this tree
        :param data: matrix of observational data that the tree was not trained on
        :return: vector of classifications
        """
        return map(self.classify, data)

    @abstractmethod
    def classify(self, observation):
        """
            Consumes an observation and outputs the classification that this tree would apply to the row
        :param observation: a row of observational data
        :return: the label that would be applied to the given row
        """
        raise NotImplementedError("EntropyTrees must implement a classify method "
                                  "that applies a prediction to an observation")
