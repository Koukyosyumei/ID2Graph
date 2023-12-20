import numpy as np
from sklearn.cluster import KMeans


class SSEMeans:
    def __init__(self, n=15, random_state=42):
        self.n = n
        self.random_state = 42

    def fit(self, x):
        sse, sse_ratio = np.zeros(self.n - 1), np.zeros(self.n - 1)
        preds = []
        for k in range(1, self.n):
            kmeans = KMeans(n_clusters=k, n_init="auto", random_state=self.random_state)
            pred = kmeans.fit_predict(x)
            preds.append(pred)
            sse[k - 1] = kmeans.inertia_

            sse_hat = (
                (self.n - k)
                / k
                * min([j + 1 / (self.n - j + 1) * sse[j] for j in range(k)])
            )
            sse_ratio[k - 1] = np.sqrt(sse[k - 1] / sse_hat)

        self.k = np.argmax(sse_ratio) + 1
        self.labels_ = preds[self.k - 1]

        return self
