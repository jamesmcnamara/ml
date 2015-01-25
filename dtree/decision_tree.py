import numpy as np
from math import log2


def lg(num):
    """
        Binary log of input, with the exception that lg(0) is 0
    :param num: number to lg
    :return: base 2 log of num
    """
    return log2(num) if num else 0


class DecisionTree:
    def __init__(self, data_store=None, eta=0, data=None, results=None, parent=None):
        if parent:
            assert data is not None and results is not None, \
                "You must pass in all three of data, results and parent or none at all " \
                "to the constructor of the decision tree"
            self.data, self.results, self.parent = data, results, parent
            self.data_store = parent.data_store
            self.eta = self.parent.eta
            self.used_columns = list(parent.used_columns)
        else:
            assert data_store is not None and eta is not 0 and data is not None and results is not None, \
                "When initializing a DecisionTree without a parent, you must pass in the DataStore and eta min"
            self.data_store = data_store
            self.eta = eta
            self.data, self.results = data, results
            self.used_columns = []
            self.parent = None
        self.split_on, self.splitter = -1, None
        self.left, self.right = None, None

        # If the number of nodes in this tree is above the eta min, the entropy is non-zero,
        # and not all columns have been used, split this tree
        if len(self.data) > self.eta and self.entropy(self.results) and len(self.used_columns) < len(self.data.T):
            self.split()
            self.classification = None
        else:
            # if len(self.data) <= self.eta:
            #     print("eta cap: {} <= {} ".format(len(self.data), self.eta))
            # elif self.entropy(self.results) == 0:
            #     print("no entropy with {} nodes".format(len(self.data)))
            # elif len(self.used_columns) == len(self.data.T):
            #     print("used up all columns")

            most_common_class, max_count = 0, 0
            for i in range(len(self.data_store.result_types)):
                if self.results.count(i) > max_count:
                    most_common_class, max_count = i, self.results.count(i)
            self.classification = most_common_class

    def test(self, data, results):
        """
            Consumes test observations and their results and returns the percentage of entries that this tree
            classified correctly
        :param data: matrix of observational data
        :param results: array of resulting data, with the indices matching the rows of data
        :return: percent of entries that were correctly classified
        """
        return sum(map(self.classify, zip(data, results))) / len(data)

    def classify(self, row_and_result):
        """
            Consumes a tuple of an observation and its corresponding result, and if this tree is a leaf,
            determines 1 if this observation was classified correctly else 0
            else uses this trees splitter to determine which sub-tree to delegate to, and then returns whether
            the subtree correctly classified the node
        :param row_and_result: a 2-tuple of observational data (1D array) and the result for that observation
        :return: 1 if this tree correctly classified the input else 0
        """
        row, result = row_and_result
        if self.splitter:
            if self.splitter(row[self.split_on]):
                return self.left.classify((row, result))
            else:
                return self.right.classify((row, result))
        else:
            return int(self.classification == result)

    def split(self):
        """
            Splits this DecisionTree on the column that provides the most information gain
            if there is more than eta nodes in this tree
        """
        self.split_on, self.splitter = self.best_attr(self.data, self.results)

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

        # Create right and left children
        self.left = DecisionTree(data=left, results=left_results, parent=self)
        self.right = DecisionTree(data=right, results=right_results, parent=self)

    def best_attr(self, data, results):
        """
            Determines the best column to split the input matrix on, and returns the column index and splitter function
        :param data: input matrix
        :param results: the classifications that correspond to the rows of the input matrix data
        :return: the a 2-tuple of column-index and a splitter function
        """
        best_info_gain, best_column, best_splitter = 0, -1, None
        for i, column in enumerate(data.T):
            if i not in self.used_columns:
                info_gain, splitter = self.best_binary_split(column, results)
                if info_gain > best_info_gain:
                    best_info_gain, best_column, best_splitter = info_gain, i, splitter
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
        info_gain = {self.information_gain(attr, results, splitter(interval)): splitter(interval)
                     for interval in sorted(attr)}

        return max(info_gain), info_gain[max(info_gain)]

    def information_gain(self, attr, results, bin_splitter):
        """
            Measures the expected reduction in entropy by splitting the results set on attr via bin_splitter
        :param attr: 1D array of attr values corresponding to the entries in results
        :param results: 1D array of classifications, with i-th classification having attribute attr[i]
        :param bin_splitter: a function that consumes an instance of the attr array and outputs a boolean indicating
            which partition to categorize the classification
        :return: Float corresponding to the information gain derived from partitioning on attr with bin_splitter
        """
        add_elements_if = lambda val: [result for element, result in zip(attr, results) if bin_splitter(element) == val]

        left, right = add_elements_if(True), add_elements_if(False)

        return self.entropy(results) - sum(len(partition) / len(attr) * self.entropy(partition)
                                           for partition in (left, right))

    def entropy(self, classifications):
        """
            Calculates the total dispersion in the input classifications by measure of the asymptotic
            bit rate transfer requirements
        :param classifications: A 1D array of classifications
        :return: float representing entropy of input set
        """
        result_dist = [0] * len(self.data_store.result_types)
        element_count = len(classifications)
        for result in classifications:
            result_dist[result] += 1

        return -sum(lg(count / element_count) * (count / element_count) for count in result_dist if count)

    def depth(self):
        """
            Determine how deep in the tree we are
        """
        return self.parent.depth() + 1 if self.parent else 0

    def __len__(self):
        """
            OVERRIDE: len(self) returns the number of entries in this node
        """
        return len(self.data)

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