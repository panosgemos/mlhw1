""" Module containing functionality regarding general data manipulation"""

import utils
import stats
import sklearn
import csv


class Data:

    def __init__(self, values=[], labels=[]):

        if len(values) != len(labels):
            raise ValueError("Values and labels must be of same length.")

        self.values = values
        self.labels = labels

    def __getitem__(self, item):
        """
        Adds indexing capabilities to the data, introducing features like
        slicing and iteration.
        :param item:
        :return:
        """
        # TODO: When one element of the dataset is selected using index
        # notation, then the values and labels of it will be integers,
        # not lists. This creates a problem because many methods in this
        # class depend on the fact that the values and labels are lists.
        return Data(self.values[item], self.labels[item])

    def __len__(self):
        return len(self.values)

    def shuffle(self, seed=None):
        """
        Shuffles the current data, using a custom provided seed.
        Otherwise, the current system time is used as a seed.

        :param seed: custom number for initializing the random generator
        """
        self.values, self.labels = sklearn.utils.shuffle(
                                self.values, self.labels, random_state=seed)

    def distance(self, instance, distfunc):
        """
        Calculates the distance between the given instance and the current
        data, measured according to the given distance function.

        :param instance: instance to calculate distance from

        :param distfunc: function for measuring the distance

        :return: the distances of the data points from the instance as a list
        """
        distances = []
        for data_instance, label in zip(self.values, self.labels):
            dist = distfunc(instance, data_instance)
            distances.append(dist)
        return distances

    def label_split(self):
        """
        Splits the data based on each different label.

        :return: a list containing the splitted data.
        """

        lbl_to_val = {}  # maps labels to corresponding list of values

        # for each values and corresponding label:
        for val, lbl in zip(self.values, self.labels):

            try:
                cur_lbl_list = lbl_to_val[lbl]
            except KeyError:    # if there is no key for the current label ...
                cur_lbl_list = []   # ... create a new empty list ...
                lbl_to_val[lbl] = cur_lbl_list  # ... and assign it to that key
            cur_lbl_list.append(val)

        splitted_data = []  # list of splitted data

        for label, values in lbl_to_val.items():
            # has copies of label equal to the number of the values list
            labels = [label] * len(values)
            splitted_data.append(Data(values, labels))

        return splitted_data

    def kfold(self, k, stratified=False, shuffle=False, seed=None):
        """
        Splits the data into k folds, and creates a generator for iterating
        over the different combinations of 1 fold for validating and k-1
        folds for training (which is k combinations in total).
        :param k: number of folds
        :param stratified: keeps the same label/class distribution in each fold
        :param shuffle: whether to shuffle the data or not
        :param seed: custom number for initializing the random generator

        :return: a generator for iterating over the different datasets created.
        """

        if shuffle:
            self.shuffle(seed)

        if stratified:

            label_separated_datasets = self.label_split()

            # list containing k-fold iterators for each of the label datasets
            label_datasets_kfold_iterators = []
            for data in label_separated_datasets:
                kfold_iterator = data.kfold(k)
                label_datasets_kfold_iterators.append(kfold_iterator)

            # iterate over each k-fold splitted dataset
            for kfold_label_datasets in zip(*label_datasets_kfold_iterators):
                dataset = Dataset.join(*kfold_label_datasets)
                if shuffle:
                    dataset.shuffle(seed)
                yield dataset

        else:

            values_iter = stats.kfold(self.values, k)
            labels_iter = stats.kfold(self.labels, k)

            for values, labels in zip(values_iter, labels_iter):
                train = Data(values[0], labels[0])
                valid = Data(values[1], labels[1])
                yield Dataset(train, valid)

    def sort_based_on(self, items):
        """
        Sorts the given items and the current data, ordered by the first one.

        :param items: items that will guide the order of the data

        :return: the sorted items.
        """

        items, self.values, self.labels = utils.sort_together(
            items, self.values, self.labels
        )
        return items

    @staticmethod
    def join(*data_series):
        """
        Joins the given series of data into one Data object.

        :param data_series: the data series to be joined
        :return: the joined Data object
        """
        values, labels = [], []
        for d in data_series:
            values.extend(d.values), labels.extend(d.labels)

        return Data(values, labels)


class Dataset:
    def __init__(self, train=Data(), valid=Data(), test=Data()):
        self.train = train
        self.valid = valid
        self.test = test

    def shuffle(self, seed=None):
        self.train.shuffle(seed)
        self.valid.shuffle(seed)
        self.test.shuffle(seed)

    @staticmethod
    def join(*datasets):
        """
        Joins the given datasets to one.

        :param datasets: datasets to be joined
        :return: a dataset created from the joined datasets
        """
        ltrain, lvalid, ltest = [], [], []

        # separate train, test and valid data by storing them to different lists
        for d in datasets:
            ltrain.append(d.train), lvalid.append(d.valid), ltest.append(d.test)

        # join the similar data together
        train, valid, test = (Data.join(*ltrain), Data.join(*lvalid),
                              Data.join(*ltest))

        return Dataset(train, valid, test)


def load_csv_data(file_path, col_types):
    """
    Loads CSV data which has the feature values first and the label/class last.

    :param file_path: path to the CSV file that holds the dataset
    :param col_types: list/tuple containing the type of the values of each CSV
                      column.

    :return: a Data object with the parsed data from the CSV file
    """
    with open(file_path) as csvfile:
        csv_iter = csv.reader(csvfile)  # get CSV parser iterator

        values = []
        labels = []

        for row in csv_iter:
            # if the row is empty, continue to the next one
            if len(row) == 0: continue
            row_values = row[:-1]   # get all but the last element (the label)
            label = row[-1]         # get the label which is the last element

            # convert the row values according to the given CSV column types
            row_values = [ctype(v) for v, ctype in zip(row_values, col_types)]
            # TECHNICAL NOTE:
            # for some really weird reason, if the row values are translated
            # into numpy arrays of 64-bit floats instead of a list of floats,
            # then only for a specific number of the dataset rows,
            # an exception is thrown in the sort_together function.
            # convert the row values into numpy arrays of 64-bit floats
            # row_values = np.array(row_values, dtype=np.float64)
            values.append(row_values)
            labels.append(label)
    return Data(values, labels)