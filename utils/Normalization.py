import numpy as np
from statistics import stdev

__author__ = 'jamesmcnamara'


def normalize(normalization, data):
    """
        For all columns with numeric type, normalize their values to be between [0,1]
        using the specified technique
    """
    if normalization == "arithmetic":
        return normalize_columns_arithmetic(data)
    elif normalization == "z-score":
        return normalize_columns_z(data)
    elif normalization == "centered":
        return center_columns(data)
    else:
        raise ValueError(normalization + " is not a normalization option")


def normalize_validation(normalization, training, validation):
    """
        For all columns with numeric type, normalize their values to be between [0,1]
        using the specified technique
    """
    if normalization == "arithmetic":
        return normalize_columns_arithmetic(validation)
    elif normalization == "z-score":
        return normalize_columns_z_validation(training, validation)
    elif normalization == "centered":
        return center_columns(validation)


def normalize_columns_arithmetic(data):
    """
        For all columns with numeric type, normalize their values to be between [0,1]
    """
    # Helper function which maps the normalization function over a column from the input matrix
    def column_mapper(col):
        minimum, maximum = min(col), max(col)
        if minimum == maximum:
            return [0.5] * len(col)
        return [(x - minimum) / (maximum - minimum) for x in col]
    return np.array([column_mapper(col) for col in data.T]).T


def normalize_columns_z(data):
    """
        For all columns with numeric type, normalize their values to be z 
        scores using z-score normalization
    """
    # Helper function which maps the normalization function over a column from the input matrix
    def column_mapper(col):
        mu = sum(col) / len(col)
        sigma = stdev(col)
        sigma = sigma if sigma else 1
        return list(map(lambda x: (x - mu) / sigma, col))
    return np.array([column_mapper(col) for col in data.T]).T


def normalize_columns_z_validation(training, validation):
    """
        For all columns with numeric type in the validation set, normalize their values to be between [0,1]
        using z-score normalization, with sigma and mu derived from the training set

    """
    # Helper function which maps the normalization function over a column from the input matrix
    def column_mapper(training_col, validation_col):
        mu = sum(training_col) / len(training_col)
        sigma = stdev(training_col)
        return list(map(lambda x: (x - mu) / sigma, validation_col))
    return np.array([column_mapper(training, validation)
                     for training, validation in zip(training.T, validation.T)]).T


def center_columns(data):
    """
        Given a matrix of data, returns that data with each column centered at 0
    :param data: input matrix
    :return: centered data
    """
    def center_col(col):
        avg = sum(col) / len(col)
        return list(map(lambda x: x - avg, col))
    return np.array(list(map(center_col, data.T))).T


def pad(design):
    """
        Consumes a design matrix and adds a column of 1's at the beginning
    :param design: a design matrix with N observations on M variables
    :return: a design matrix with N observations on M+1 variables
    """
    h, w = design.shape
    padded = np.ones((h, w + 1))
    padded[:, 1:] = design
    return padded

