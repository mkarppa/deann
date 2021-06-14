# This is a modified copy of this file:
# https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/algorithms/faiss.py
#

import sys
import numpy as np
import sklearn.preprocessing
import faiss

from extern.base import BaseANN


class Faiss(BaseANN):
    def query(self, v, n):
        s_before = self.get_additional()['dist_comps']
        if self._metric == 'angular':
            v /= np.linalg.norm(v)
        D, I = self.index.search(np.expand_dims(
            v, axis=0).astype(np.float32), n)
        s_after = self.get_additional()['dist_comps']
        return np.sqrt(D[0]), I[0], np.array([s_after-s_before], dtype=np.int32)

    def batch_query(self, X, n):
        if self._metric == 'angular':
            X /= np.linalg.norm(X)
        self.res = self.index.search(X.astype(np.float32), n)

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res


    
class FaissIVF(Faiss):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        index = faiss.IndexIVFFlat(
            self.quantizer, X.shape[1], self._n_list, faiss.METRIC_L2)
        index.train(X)
        index.add(X)
        self.index = index

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis +      # noqa
                faiss.cvar.indexIVF_stats.nq * self._n_list}

    def __str__(self):
        return 'FaissIVF(n_list=%d, n_probe=%d)' % (self._n_list,
                                                    self._n_probe)
