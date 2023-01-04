from sklearn import metrics, preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance


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
