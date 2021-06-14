from sklearn.neighbors import NearestNeighbors
from extern.base import BaseANN
import numpy as np

class BruteNN(BaseANN):
    def __init__(self, metric = 'euclidean', return_dists = True,
                     return_samples = True):
        if metric == 'euclidean':
            self._metric = 'euclidean'
        elif metric == 'taxicab':
            self._metric = 'manhattan'
        else:
            raise ValueError(f'invalid metric ``{metric}\'\' supplied')
        self._return_dists = return_dists
        self._return_samples = return_samples
    
    def fit(self, X):
        self._nn = NearestNeighbors(algorithm='brute', metric=self._metric)
        self._nn.fit(X)
        self._n = X.shape[0]

    def query(self, q, n):
        if self._return_dists and self._return_samples:
            return self._nn.kneighbors(q.reshape(1,-1),n) + (np.array([self._n], np.int32),)
        elif self._return_dists and not self._return_samples:
            return self._nn.kneighbors(q.reshape(1,-1),n)
        elif not self._return_dists and self._return_samples:
            return (self._nn.kneighbors(q.reshape(1,-1),n)[1], np.array([self._n], np.int32))
        else:
            return self._nn.kneighbors(q.reshape(1,-1),n)[1]
