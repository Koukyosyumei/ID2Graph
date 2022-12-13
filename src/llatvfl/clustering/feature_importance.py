from sklearn import metrics, preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance


class KMeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=2, random_state=42):
        self.n_classes = n_classes
        self.random_state = random_state

    def fit(self, X, y):
        self.min_max_scaler = preprocessing.MinMaxScaler()
        X = self.min_max_scaler.fit_transform(X)
        self.km = KMeans(n_clusters=self.n_classes, random_state=self.random_state).fit(
            X
        )
        return self

    def predict(self, X):
        return self.km.predict(self.min_max_scaler.transform(X))

    def score(self, X, y):
        return metrics.v_measure_score(y, self.predict(X))


def calculate_permutation_importance_for_kmeans_clustering(
    X, y, n_repeat=30, random_state=42
):
    kmc = KMeansClassifier()
    result = permutation_importance(
        kmc, X, y, n_repeats=n_repeat, random_state=random_state
    )

    return result["importances_mean"]
