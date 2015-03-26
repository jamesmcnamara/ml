import numpy as np
from nose.tools import assert_almost_equal, assert_equal

from text_classification.NaiveBayes import NaiveBayes

__author__ = 'jamesmcnamara'

data = np.array([[1, 1, 0, 0, 1],   # 1
                 [1, 1, 1, 0, 0],   # 1
                 [1, 1, 0, 1, 1],   # 1

                 [1, 0, 0, 0, 1],   # 2
                 [0, 1, 1, 1, 1],   # 2
                 [1, 0, 1, 0, 1],   # 2

                 [0, 1, 0, 0, 1],   # 3
                 [1, 0, 0, 1, 1],   # 3
                 [0, 0, 0, 0, 1]])  # 3
                # 6  5  3  3  8

mata = np.array([[1, 2, 0],   # 1
                 [1, 3, 5],   # 1

                 [1, 0, 0],   # 2
                 [0, 2, 5],   # 2

                 [9, 7, 6],   # 3
                 [1, 3, 2]])  # 3
                # 13 17 18
                
breakpoints = [i * 3 for i in range(4)]
mreakpoints = [i * 2 for i in range(4)]

by_label = [data[breakpoints[i]:breakpoints[i+1]] for i in range(3)]
my_label = [mata[mreakpoints[i]:mreakpoints[i+1]] for i in range(3)]


def test_cond_probs():
    cond_probs = NaiveBayes.get_cond_probs_static(3, data, breakpoints, False)
    for label, label_predictions in enumerate(cond_probs):
        for word_id, prob in enumerate(label_predictions):
            total = len(by_label[label]) + 2
            in_class = sum(by_label[label][:, word_id]) + 1
            assert_almost_equal(prob, in_class / total)


def test_cond_prob_bern():
    one_probs = NaiveBayes.get_class_prob(by_label[0], False)
    one_expect = [4/5, 4/5, 2/5, 2/5, 3/5]
    two_probs = NaiveBayes.get_class_prob(by_label[1], False)
    two_expect = [3/5, 2/5, 3/5, 2/5, 4/5]
    three_probs = NaiveBayes.get_class_prob(by_label[2], False)
    three_expect = [2/5, 2/5, 1/5, 2/5, 4/5]

    for actual, expected in zip(one_probs, one_expect):
        assert_almost_equal(actual, expected)

    for actual, expected in zip(two_probs, two_expect):
        assert_almost_equal(actual, expected)

    for actual, expected in zip(three_probs, three_expect):
        assert_almost_equal(actual, expected)


def test_cond_probs_multi():
    one_probs = NaiveBayes.get_class_prob(my_label[0], True)
    one_expect = [3/14, 6/14, 6/14]
    two_probs = NaiveBayes.get_class_prob(my_label[1], True)
    two_expect = [2/10, 3/10, 6/10]
    three_probs = NaiveBayes.get_class_prob(my_label[2], True)
    three_expect = [11/30, 11/30, 9/30]

    for actual, expected in zip(one_probs, one_expect):
        assert_almost_equal(actual, expected)

    for actual, expected in zip(two_probs, two_expect):
        assert_almost_equal(actual, expected)

    for actual, expected in zip(three_probs, three_expect):
        assert_almost_equal(actual, expected)


def test_classify_bern():
    docs = [[1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1]]
    predictions = [2, 3, 2]
    cond_probs = NaiveBayes.get_cond_probs_static(3, data, breakpoints, False)
    class_priors = [1/3, 1/3, 1/3]
    for doc, pred_label in zip(docs, predictions):
        classification = NaiveBayes.classify(doc, cond_probs, class_priors, 3, False)
        assert_equal(classification, pred_label)


def test_classify_multi():
    docs = [[3, 6, 6],
            [2, 3, 6],
            [11, 11, 9]]
    predictions = [1, 2, 3]
    cond_probs = NaiveBayes.get_cond_probs_static(3, mata, mreakpoints, True)
    class_priors = [1/3, 1/3, 1/3]
    for doc, pred_label in zip(docs, predictions):
        classification = NaiveBayes.classify(doc, cond_probs, class_priors, 3, True)
        assert_equal(classification, pred_label)