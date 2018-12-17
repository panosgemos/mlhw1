from matplotlib import pyplot as plt

from data import *
from ml.knn import *


def main():
    # iris dataset is made of 4 features of type float and a label of type str
    iris_col_types = [float]*4 + [str]
    iris = load_csv_data('ds/iris.data', iris_col_types)

    # pima dataset is made of 8 features of type float and a label of type int
    pima_col_types = [float]*8 + [int]
    pima = load_csv_data('ds/pima-indians-diabetes.data', pima_col_types)

    folds = 10
    seed = 1
    max_neighbors = 100
    odd = True

    iris_avg_acc, iris_neighbors = multi_cv_knn(iris, folds, max_neighbors,
                                                seed, odd)
    pima_avg_acc, pima_neighbors = multi_cv_knn(pima, folds, max_neighbors,
                                                seed, odd)

    print("Iris average accuracy: {}%".format(iris_avg_acc))
    print("Pima average accuracy: {}%".format(pima_avg_acc))

    plt.figure(1)
    plt.plot(iris_neighbors, iris_avg_acc, 'C0', label='iris')
    plt.xlabel('neighbors')
    plt.ylabel('accuracies')
    plt.title("[Iris] K-NN with 10-Fold Cross Validation")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(pima_neighbors, pima_avg_acc, 'C1', label='pima')
    plt.xlabel('neighbors')
    plt.ylabel('accuracies')
    plt.title("[Pima Indians Diabetes] K-NN with 10-Fold Cross Validation")
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(iris_neighbors, iris_avg_acc, label='iris')
    plt.plot(pima_neighbors, pima_avg_acc, label='pima')
    plt.xlabel('neighbors')
    plt.ylabel('accuracies')
    plt.title("[Both Datasets] K-NN with 10-Fold Cross Validation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
