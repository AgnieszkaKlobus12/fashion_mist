import numpy as np


# returns matrix of distances between objects in both lists
def hamming_distance(x, x_train):
    x = np.reshape(x, (len(x), 784))
    x_train = np.reshape(x_train, (len(x_train), 784))
    ones1 = np.ones_like(x)
    tran1 = np.transpose(x_train)
    elem1 = ones1 - x
    elem1 = elem1 @ tran1

    ones2 = np.ones_like(x_train)
    tran2 = np.transpose(ones2 - x_train)
    elem2 = x @ tran2

    result = elem1 + elem2

    return result
    pass


# returns sorted labels matrix
def sort_train_labels(dist, labels):
    w = dist.argsort(kind='mergesort')
    result = labels[w]
    return result
    pass


# returns matrix of probabilities
def probabilities_matrix(labels_matrix, classes):
    result = []
    for i in range(len(labels_matrix)):
        row = []
        for el in range(classes):
            row.append(labels_matrix[i][el])
        count = np.bincount(row, None, 10)
        result.append(
            [count[0] / classes, count[1] / classes, count[2] / classes, count[3] / classes, count[4] / classes,
             count[5] / classes, count[6] / classes,
             count[7] / classes, count[8] / classes, count[9] / classes])
    return result
    pass


def classification_error(probabilities, true_labels):
    diff = 0
    for row in range(len(probabilities)):
        if (np.argmax(probabilities[row])) != true_labels[row]:
            diff += 1
    return diff / len(true_labels)
    pass


def probabilities_for_k(x_train, x_test, y_train, y_test, k):
    dist = hamming_distance(x_test, x_train)
    sorted_labels = sort_train_labels(dist, y_train)

    results = probabilities_matrix(sorted_labels, k)
    error = classification_error(results, y_test)
    return results, error


def model_selection(x_train, x_val, y_val, y_train, k_values):
    dist = hamming_distance(x_val, x_train)
    sorted_labels = sort_train_labels(dist, y_train)

    errors = []
    for k in range(len(k_values)):
        errors.append(classification_error(probabilities_matrix(sorted_labels, k_values[k]), y_val))

    best_error = min(errors)
    best_k = k_values[np.argmin(errors)]
    return best_error, best_k, errors
