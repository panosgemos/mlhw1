""" Python module for the bayes algorithm """

# TODO: This is a draft module to be implemented.

import numpy as np

from ml import knn


def main():
    col_types = [float]*8 + [int]
    dataset = knn.load_csv_dataset('pima-indians-diabetes.data', col_types)
    dataset.sort_based_on(dataset.labels)
    for v, l in zip(dataset.values, dataset.labels):
        print(v, l)

    dataset0, dataset1 = dataset[:500], dataset[500:]

    mean0 = np.empty([8,1])
    mean1 = np.empty([8,1])
    for i in range(0, 8):
        a = np.array(dataset0.values, dtype=np.float)
        b = np.array(dataset1.values, dtype=np.float)
        mean0[i] = np.mean(a[:, i])
        mean1[i] = np.mean(b[:, i])

    S = np.diag([3.4, 32.0, 19.4, 16.0, 115.2, 7.9, 0.3, 11.8])
    print(S)
    # mean0 = np.mean(dataset0.values)
    # mean1 = np.mean(dataset1.values)

    print(mean0, mean1)

main()