import numpy as np
from math import log
__author__ = 'jamesmcnamara'


class NaiveBayes:
    def __init__(self, datastore):
        self.datastore = datastore
        self.multi = datastore.model == "multinomial"
        self.data = datastore.data
        self.labels = datastore.data_labels
        self.label_count = self.labels[-1]
        self.vocab = datastore.vocab
        self.breakpoints = datastore.breakpoints
        self.class_priors = [self.labels.count(i + 1)/len(datastore.data_labels)
                             for i in range(self.label_count)]

        self.cond_probs = self.get_cond_probs()

    def get_cond_probs(self):
        """
            Returns the matrix of conditional probabilities for each class by each word.
             Wrapper for get_cond_probs_static
        :return: class by word matrix of the conditional probability where the probability of word i appearing in a
            document given that the document is from class j is given by result[j, i]
        """
        return self.get_cond_probs_static(self.label_count, self.data, self.breakpoints, self.multi)

    def predict(self, data):
        """
            returns a stream of predicted classifications for the given data. Wrapper for predict_static
        :param data: matrix of documents by the words that appear in them (either 1 or count)
        :return: stream of predictions
        """
        return self.predict_static(data, self.cond_probs, self.class_priors, self.label_count, self.multi)

    @staticmethod
    def get_cond_probs_static(label_count, data, breakpoints, multi):
        """
            Returns the matrix of conditional probabilities for each class by each word.
        :param label_count: number of labels that exist in this universe of classification
        :param data: training data (documents by words)
        :param breakpoints: list describing where labels start and end in training data, such that
            data[breakpoints[3]:breakpoints[4]] returns all of the documents of class 3
        :param multi: Boolean determining multinomial or multivariate bernoulli
        :return: class by word matrix of the conditional probability where the probability of word i appearing in a
            document given that the document is from class j is given by result[j, i]
        """
        return np.array([NaiveBayes.get_class_prob(data[breakpoints[c]:breakpoints[c + 1]], multi)
                         for c in range(label_count)])

    @staticmethod
    def get_class_prob(label_block, multi):
        """
            Given a block of documents from a single class, returns the conditional probability of each word appearing
        :param label_block: documents * words
        :param multi: Boolean determining multinomial or multivariate bernoulli
        :return: a list where the ith element denotes the probability of the ith word appearing
        """
        denominator = sum(sum(label_block)) if multi else len(label_block)
        return [(sum(label_block[:, w]) + 1) / (denominator + 2) for w in range(len(label_block.T))]

    @staticmethod
    def predict_static(data, cond_probs, class_priors, label_count, multi):
        """
            Returns a stream of predicted classifications for the given data
        :param data: documents * words
        :param cond_probs: class by word matrix of the conditional probability where the probability of word i appearing in a
            document given that the document is from class j is given by result[j, i]
        :param class_priors: list where the ith element denotes the proportion of the ith class in the training set
        :param label_count: number of labels that exist in this universe of classification
        :param multi: Boolean determining multinomial or multivariate bernoulli
        :return: stream of predictions
        """
        classify = lambda doc: NaiveBayes.classify(doc, cond_probs, class_priors, label_count, multi)
        return map(classify, data)

    @staticmethod
    def classify(document, cond_probs, class_priors, label_count, multi):
        """
            Classifies the given document as belonging to one of the possible labels
        :param document: a vector of word frequencies or observations
        :param cond_probs: class by word matrix of the conditional probability where the probability of word i appearing in a
            document given that the document is from class j is given by result[j, i]
        :param class_priors: list where the ith element denotes the proportion of the ith class in the training set
        :param label_count: number of labels that exist in this universe of classification
        :param multi: Boolean determining multinomial or multivariate bernoulli
        :return: label for the given document
        """
        predictor, balance = NaiveBayes._classification_functions(multi)

        probabilities = np.array([log(class_priors[i])
                                 + balance(document)
                                 + sum(map(predictor, cond_probs[i], document))
                                 for i in range(label_count)])

        return np.argmax(probabilities) + 1

    @staticmethod
    def _classification_functions(multi):
        """
            generates the prediction and balancing functions for the multinomial
            or multivariate bernoulli distributions
        :param multi: multinomial or multivariate bernoulli distributions
        :return: prediction function and balancing function tuple
        """
        if multi:
            # uses log probability to circumvent float overflow on factorial conversion
            lg_fact = lambda n: sum(map(log, range(1, n + 1)))
            predictor = lambda p, x: x * log(p) - lg_fact(x)
            balance = lambda n: lg_fact(sum(n))
        else:
            predictor = lambda p, x: p if x else 1 - p
            balance = lambda _: 1

        return predictor, balance


