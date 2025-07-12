from __future__ import print_function
import numpy as np
from nltk.cluster import euclidean_distance


class KNN:
    def __init__(self, k=4):
        self.k = k

    def _vote(self, neighbor_labels):
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()

    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])

        for i, test_sample in enumerate(X_test):
            idx = np.argsort([euclidean_distance(test_sample, x) for x in X_train])
            knn = np.array([y_train[i] for i in idx])
            y_pred[i] = self._vote(knn)

        return y_pred
