""" Module that contains functions and classes regarding statistics """

import utils

from itertools import chain
import numpy as np


def kfold(data, k):
    """
    Splits the given data into k folds and creates two sets for each
    different combination of the k-1 folds for the training set and 1 fold
    for the test set.

    :param data: data from which the folds will be created
    :param k: number of folds

    :return: a generator for iterating over the different training and test data
    """

    folds = utils.partition(data, k)

    for i in range(k):
        test = folds[i]
        train_folds = chain(folds[0:i], folds[i+1:k])
        train = [item for fold in train_folds for item in fold]
        yield train, test

class CrossValid:

    def __init__(self, data, folds, shuffle=False, seed=None):
        self.data = data
        self.folds = folds
        self.shuffle = shuffle
        self.seed = seed

    def run(self, algorithm, **algorithm_parameters):
        accuracies = []

        # generate predictions

        for dataset in self.data.kfold(self.folds, shuffle=self.shuffle,
                                        seed=self.seed):
            accuracy = algorithm(dataset, **algorithm_parameters)
            accuracies.append(accuracy)

        average_accuracy = np.average(np.array(accuracies, dtype=np.float64))

        # print("Average accuracy: {}%".format(average_accuracy))

        return average_accuracy
