import deann
import numpy as np
import sys
sys.path.append("..")
from extern.faiss import FaissIVF

# dataset available here: https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
X = np.loadtxt('shuttle.trn', dtype = np.float64)[:,:-1].copy()
Q = np.loadtxt('shuttle.tst', dtype = np.float64)[:,:-1].copy()

n, d = X.shape
h = 16.0 # the bandwidth

nkde = deann.NaiveKde(h, 'exponential')
nkde.fit(X)
mu, _ = nkde.query(Q)
idx = mu > 0

fivf = FaissIVF('euclidean', 1024)
fivf.fit(X)
fivf.set_query_arguments(5)

k = 100
m = 1000
annp = deann.AnnEstimatorPermuted(h, 'exponential', k, m, fivf)
annp.fit(X)
Z, S = annp.query(Q)
print(np.mean(S))
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))