# Module containing machine learning algorithms

import operator
import numpy as np
from stats import CrossValid


def euclidean(instance1, instance2):
    distance = 0
    for comp1, comp2 in zip(instance1, instance2):
        distance += (comp1 - comp2) ** 2
    return np.sqrt(distance)


def get_neighbors(training_set, test_instance, k):
    distances = training_set.distance(test_instance, euclidean)
    training_set.sort_based_on(distances)
    # get the first k instances from the dataset, that is the first k neighbors
    neighbors = training_set[0:k]
    return neighbors


def get_response(neighbors):
    class_votes = {}
    for neighbor_label in neighbors.labels:
        if neighbor_label in class_votes:
            class_votes[neighbor_label] += 1
        else:
            class_votes[neighbor_label] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1),
                         reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    correct = 0
    for test_label, prediction in zip(test_set.labels, predictions):
        if test_label == prediction:
            correct += 1
    # test_set_size = np.array(len(test_set), dtype=np.float64)
    return correct / len(test_set) * 100.0


def knn(dataset, k):
    predictions = []
    for valid_instance, valid_label in zip(dataset.valid.values,
                                           dataset.valid.labels):
        neighbors = get_neighbors(dataset.train, valid_instance, k)
        result = get_response(neighbors)
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(
        #     valid_label))
    accuracy = get_accuracy(dataset.valid, predictions)
    # print('Accuracy: ' + repr(accuracy) + '%')
    return accuracy





def multi_cv_knn(data, folds, max_neighbors, seed=None, odd=False):
    avg_accuracies = []

    step = 2 if odd else 1

    neighbors = list(range(1, max_neighbors + 1, step))

    for k in neighbors:
        acc = CrossValid(data, folds, shuffle=True, seed=seed).run(knn, k=k)
        avg_accuracies.append(acc)
        print(".", end="", flush=True)  # print progress

    print(" Done")

    return avg_accuracies, neighbors
