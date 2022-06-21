import copy
import random

import numpy as np


class ReducedKMeans:
    """Implementatino of Reduced Kmeans

    Examples:
        >>from sklearn import datasets
        >>from matplotlib import pyplot as plt
        >>iris = datasets.load_iris()
        >>rkm = ReducedKMeans(3, n_dim=2)
        >>rkm.fit(iris.data)
        >>plt.scatter(rkm.X_projected[:, 0], rkm.X_projected[:, 1], c=rkm.labels_)
    """

    def __init__(
        self,
        n_clusters=2,
        n_dim=2,
        n_init=10,
        max_itr=300,
        tol=1e-4,
        verbose=0,
        random_state=0,
    ):
        self.n_dim = n_dim
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_itr = max_itr
        self.tol = tol
        self.verbose = verbose

        random.seed(random_state)
        np.random.seed(random_state)

    def fit(self, X):
        n_row, n_col = X.shape
        self.E_hat = np.zeros((n_row, self.n_clusters))
        self.C_hat = np.zeros((self.n_clusters, n_col))
        loss_best = np.Inf
        self.loss_logs = []

        count = 0
        while count < self.n_init:
            loss = np.Inf

            # initialize the cluster mean
            idx = np.random.choice(np.arange(n_row), self.n_clusters, replace=False)
            Y = X[idx, :]  # n_clusters \times n_col
            C = np.copy(Y)  # centroid matrix, n_clusters \times n_col

            inner_count = 0
            losses = []
            while inner_count < self.max_itr:

                # E-step: assign objects to corresponding cluster
                E = np.zeros((n_row, self.n_clusters))
                for i in range(n_row):
                    dist_list = [
                        np.sum((X[i, :] - C[k, :]) ** 2) for k in range(self.n_clusters)
                    ]
                    min_idx = np.argmin(dist_list)
                    E[i, min_idx] = 1

                W = np.dot(E.T, E)
                ns = np.diag(W)
                Y = (np.dot(X.T, E) / ns).T

                # C-step
                w_sqrt_Y = np.dot(np.diag(np.sqrt(ns)), Y)
                U, S, V = np.linalg.svd(w_sqrt_Y, full_matrices=False)
                C = np.dot(
                    np.diag(np.sqrt(1 / ns)),
                    np.dot(
                        U[:, : self.n_dim],
                        np.dot(np.diag(S[: self.n_dim]), V[: self.n_dim, :]),
                    ),
                )

                # evaluation
                loss_new = np.trace(np.dot((X - np.dot(E, C)).T, (X - np.dot(E, C))))
                losses.append(loss_new)
                diff = loss - loss_new
                if diff < self.tol:
                    break
                loss = loss_new

                if self.verbose > 0 and inner_count % self.verbose == 0:
                    print(f"trial={count+1}, round={inner_count+1}: {loss}")

                inner_count += 1

            if loss_new < loss_best:
                loss_best = loss_new
                self.loss_logs = copy.deepcopy(losses)
                self.C_hat = C
                self.E_hat = E

            count += 1

        U, S, V = np.linalg.svd(self.C_hat, full_matrices=False)
        self.A = U[:, : self.n_dim]
        self.B = np.dot(np.diag(S[: self.n_dim]), V[: self.n_dim, :]).T
        self.X_projected = np.dot(
            X, np.dot(V[: self.n_dim, :].T, np.diag(1 / S[: self.n_dim]))
        )
        self.labels_ = np.where(self.E_hat == 1)[1]

        return self

    def predict(self, X_test):
        n_row = X_test.shape[0]
        E = np.zeros((n_row, self.n_clusters))
        for i in range(n_row):
            dist_list = [
                np.sum((X_test[i, :] - self.C_hat[k, :]) ** 2)
                for k in range(self.n_clusters)
            ]
            min_idx = np.argmin(dist_list)
            E[i, min_idx] = 1
        predicted_labels = np.where(E == 1)[1]
        return predicted_labels
