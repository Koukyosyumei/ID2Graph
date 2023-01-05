import numpy as np
from sklearn import metrics, preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from tqdm import tqdm


class KMeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=2, random_state=42):
        self.n_classes = n_classes
        self.random_state = random_state

    def fit(self, X, y):
        self.km = KMeans(n_clusters=self.n_classes, random_state=self.random_state)
        self.km.fit(X)
        return self

    def predict(self, X):
        return self.km.predict(X)

    def score(self, X, y):
        return metrics.v_measure_score(y, self.predict(X))


def calculate_permutation_importance_for_kmeans_clustering(
    X_train, y_train, n_classes=2, n_repeat=15, random_state=42
):
    mm = preprocessing.MinMaxScaler()
    X_minmax = mm.fit_transform(X_train)
    kmc = KMeansClassifier(n_classes=n_classes, random_state=random_state)
    kmc.fit(X_minmax, y_train)
    print("Calculating feature importance ...")
    result = permutation_importance(
        kmc, X_minmax, y_train, n_repeats=n_repeat, random_state=random_state
    )

    return result["importances_mean"]


def drop_column_importance(X_train, y_train):
    mm = preprocessing.MinMaxScaler()
    X_train = mm.fit_transform(X_train)

    num_col = X_train.shape[1]
    clf = KMeansClassifier(n_classes=len(np.unique(y_train)))
    base_score = clf.fit(X_train, y_train).score(X_train, y_train)

    imp = np.zeros(num_col)
    for i in tqdm(range(num_col)):
        clf.fit(X_train[:, [j for j in range(num_col) if j != i]], y_train)
        imp[i] = (
            clf.score(X_train[:, [j for j in range(num_col) if j != i]], y_train)
            - base_score
        )

    return imp


def order_importance(X_train, y_train):
    num_col = X_train.shape[1]
    best_order = []
    clf = KMeansClassifier(n_classes=len(np.unique(y_train)))
    used_col = set()

    pbar = tqdm(total=num_col)

    while len(best_order) != num_col:
        best_score = -1e6
        for i in range(num_col):
            if i not in used_col:
                clf.fit(X_train[:, best_order + [i]], y_train)
                tmp_score = clf.score(X_train[:, best_order + [i]], y_train)
                if tmp_score > best_score:
                    best_score = tmp_score
                    best_col = i

        used_col.add(best_col)
        best_order.append(best_col)
        pbar.update(1)

    pbar.close()

    fti = np.zeros(num_col)
    for i in range(num_col):
        fti[best_order[i]] = num_col - i

    return fti
