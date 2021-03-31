import pandas as pd
import numpy as np
import os
import sys


def load_file(file_path):
    data = pd.read_csv(file_path, header=None)

    return data


def preprocess_data(data):
    data = np.expand_dims(data, axis=-1)
    return data


def load_data(
        train_paths: list,
        test_paths: list
):
    train_data = pd.concat([load_file(train_file) for train_file in train_paths]).values
    train_X, train_Y = train_data[:, :-1], train_data[:, -1]

    test_data = pd.concat([load_file(test_file) for test_file in test_paths]).values
    test_X, test_Y = test_data[:, :-1], test_data[:, -1]

    return (preprocess_data(train_X), train_Y), (preprocess_data(test_X), test_Y)


# (train_X, train_Y), (test_X, test_Y) = load_fold(
#     [os.path.join(os.getcwd(), "../data/input.fold.train1.csv")],
#     [os.path.join(os.getcwd(), "../data/input.fold.test1.csv")]
# )
#
# print(train_X.shape)
# print(test_Y.shape)

