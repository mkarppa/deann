import deann
import numpy as np

# dataset available here: https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
# load the data
# the last column is for class label which we want to disregard
# a copy must be made to ensure that the data is contiguously in memory
X = np.loadtxt('shuttle.trn', dtype = np.float64)[:,:-1].copy()
Q = np.loadtxt('shuttle.tst', dtype = np.float64)[:,:-1].copy()

n, d = X.shape

h = 16.0 # the bandwidth

nkde = deann.NaiveKde(h, 'exponential')
nkde.fit(X)
(Z, S) = nkde.query(Q)
print(np.median(Z))
median_samples = np.median(S)
print(median_samples)
print(np.all(S == median_samples))

