from dtree.decision_tree import BinaryTree

__author__ = 'jamesmcnamara'


class RegressionTree(BinaryTree):
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
            Consumes a list of numbers and returns a generator which yields the average forever
        :param results: 1D array of integers
        :return: iterator which always yields the average of the input set
        """
        avg = sum(results) / len(results)
        while True:
            yield avg

    def value(self):
        """
            Classifies nodes in this tree by the average of the results data set
        :return: Average value of the classifications for this tree
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
        return self.mean_squared_error(results, (self.regress(row) for row in data))

    def regress(self, row):
        """
            Classifies the given row and
        :param row:
        :return: classification
        """
        if self.split_on.splitter:
            if self.split_on.splitter(row[self.split_on.column]):
                return self.left.regress(row)
            else:
                return self.right.regress(row)
        else:
            return self.value()

