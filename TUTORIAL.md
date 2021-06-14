# DEANN Tutorial

This tutorial describes very concisely how to use the Python module. We will be using the [Shuttle dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)) as a simple and small example.

## A word about threading

For consistent single-threaded runtimes, you should disable multithreading. This can be achieved by setting the following environment variables:
```
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
```
Alternatively, you may disable threading within a Python script as long as you do this *before* you load `deann` or any other module that depends on MKL:
```
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import deann
```

## Computing the estimate naively

Let us start by loading the data vectors, and computing the exact KDE values. 
```
import deann
import numpy as np

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
```

Expected output:
```
0.13927534820259496
43500.0
True
```

This shows the basic interface of DEANN: first construct an object by setting the bandwidth, choosing the kernel, and possibly providing other arguments, then fit the data (in the case of Naive KDE, this amounts to only storing a reference to the data), and finally perform a query.

The query returns a 2-tuple: the Z contains the actual KDE values, and S records the number of samples looked at per query. In this example, we see that every sample was looked at in each query.

## Comparing to scikit-learn

We will then compare these with the values obtained by using scikit-learn to check that the values are in agreement. Because DEANN does not perform the normalization that scikit-learn does (that is, divide by a constant such that the kernel function integrates to unity), we'll just correct for that; also, scikit-learn returns the logarithm of the estimate.
```
from sklearn.neighbors import KernelDensity
sklearn_kde = KernelDensity(bandwidth = h, kernel = 'exponential')
sklearn_kde.fit(np.zeros((1,d)))
# logarithmic normalization constant
C = sklearn_kde.score_samples(np.zeros((1,d))) 

sklearn_kde.fit(X)
mu = np.exp(sklearn_kde.score_samples(Q)-C) # note the C
print(np.amax(np.abs(mu-Z)))
```
Expected output:
```
9.134359935103475e-14
```

## Other kernels

We can also try other kernels. The default kernel `'exponential'` corresponds to the function K_h(x,y)=exp(-||x-y||_2/h). The other supported kernels are `'gaussian'` and `'laplacian`' which correspond to the functions K_h(x,y)=exp(-||x-y||_2^2/2/h^2) and K_h(x,y)=exp(-||x-y||_1/h), respectively.
```
nkde = deann.NaiveKde(h, 'gaussian')
nkde.fit(X)
(Z, S) = nkde.query(Q)
print(np.median(Z))

sklearn_kde = KernelDensity(bandwidth = h, kernel = 'gaussian')
sklearn_kde.fit(np.zeros((1,d)))
C = sklearn_kde.score_samples(np.zeros((1,d))) 

sklearn_kde.fit(X)
mu = np.exp(sklearn_kde.score_samples(Q)-C)
print(np.amax(np.abs(mu-Z)))

nkde = deann.NaiveKde(h, 'laplacian')
nkde.fit(X)
(Z, S) = nkde.query(Q)
print(np.median(Z))

sklearn_kde = KernelDensity(bandwidth = h, kernel = 'exponential', metric = 'manhattan')
sklearn_kde.fit(np.zeros((1,d)))
C = sklearn_kde.score_samples(np.zeros((1,d))) 

sklearn_kde.fit(X)
mu = np.exp(sklearn_kde.score_samples(Q)-C)
print(np.amax(np.abs(mu-Z)))
```
Expected output:
```
0.17597973927696575
2.82052159406021e-13
0.029616495513893566
8.409939411535561e-14
```

## Random Sampling

Let us have a look at the Random Sampling estimator.
```
m = 500 # number of random samples
seed = 1234 # random number seed, optional
rs = deann.RandomSampling(h, 'exponential', m, seed)
rs.fit(X)
(Z, S) = rs.query(Q)
print(np.all(S == m))
```
Expected output:
```
True
```

The RS estimator has one parameter m for the number of samples. This is confirmed by the output: we have looked at exactly m samples per query.

The seed parameter is only used for initializing the pseudorandom number generator, and to make the runs replicable. The seed parameter is found for all other randomized estimators as well and can be omitted.

Let's see how the *relative* error looks.
```
nkde = deann.NaiveKde(h, 'exponential')
nkde.fit(X)
mu, _ = nkde.query(Q)

# the data contains outliers whose KDEs are identically 0, so we remove these
idx = mu > 0
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))
```
Expected output:
```
0.059753296457375434
```

With all estimators that have parameters, we can use the function `reset_parameters` to reset the parameters. Let's see what happens if we increase the number of random samples.

```
rs.reset_parameters(5000)
Z, _ = rs.query(Q)
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))
```
Expected output:
```
0.01909539440513196
```
So the larger the m, the better the estimate.

Let's see what the random number seed does.
```
rs.reset_parameters(500)
rs.reset_seed(1234)
Z, _ = rs.query(Q)
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))

rs.reset_seed(1234)
Z, _ = rs.query(Q)
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))

rs.reset_seed()
Z, _ = rs.query(Q)
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))

rs = deann.RandomSampling(h, 'exponential', m)
rs.fit(X)
Z, _ = rs.query(Q)
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))
```
Expected output:
```
0.059753296457375434
0.059753296457375434
0.057553529521559496
0.0592363464140487
```
So resetting the seed with `reset_seed` can make the runs predictable. Leaving the seed out causes unpredictable initialization, also if the seed is left out from the constructor call.

## Permuted Random Sampling
Let's compare `RandomSampling` with `RandomSamplingPermuted`. This estimator does not perform sampling during queries, but during initialization, and uses matrix multiplication to accelerate the queries. `RandomSamplingPermuted` only supports Euclidean kernels, that is, `'exponential'` and `'gaussian'`, and the number of samples must not exceed the size of the dataset.

```
rsp = deann.RandomSamplingPermuted(h, 'exponential', m, seed)
rsp.fit(X)
(Z, S) = rsp.query(Q)
print(np.all(S == m))
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))

rsp.reset_parameters(5000)
Z, _ = rsp.query(Q)
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))
```
Expected output:
```
True
0.06323608496860611
0.018722107915015274
```
Notice that the estimator behaves very similarly to `RandomSampling`. However, `RandomSamplingPermuted` uses sampling without repetition. In particular, if we take a sample of the size n, the estimator computes the values exactly:
```
rsp.reset_parameters(n)
(Z, S) = rsp.query(Q)
print(np.all(S == n))
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))
```
Expected output:
```
True
7.020564178728595e-15
```

The estimator object maintains a permuted copy of the dataset and a pointer to the start of the next contiguous sample. While the random number generator can be reset with `reset_seed`, this does not repermute the data or reset the pointer as the repermutation is an expensive operation. If repermutation is desired, the dataset must be refit.
```
rsp.reset_parameters(m)
rsp.reset_seed(seed)
rsp.fit(X)
Z, _ = rsp.query(Q)
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))
```
Expected output:
```
0.06323608496860611
```

## The AnnEstimator

We now turn our attention to the actual DEANN estimator that uses ANN objects for choosing nearby points whose contribution is computed exactly. The estimator comes in two varieties: `AnnEstimator` and `AnnEstimatorPermuted`. The difference is in the way random sampling is performed: `AnnEstimator` uses vanilla random sampling with replacement, but `AnnEstimatorPermuted` uses permuted random sampling without replacement. The differences between the two are the same as those between `RandomSampling` and `RandomSamplingPermuted`: the latter holds a permuted copy of the dataset, and resetting the random seed requires refitting of the data; furthermore, the permuted variant only works for Euclidean kernels. For the remainder of this tutorial, we will be looking at `AnnEstimator`, but the examples should work without any change for `AnnEstimatorPermuted` as well.

The DEANN estimator has three parameters: the number of neighbors to query k, the number of random samples to draw m, and the ANN object A. The first two integer parameters are self-explanatory, and A is a Python object that conforms to a certain interface. We will postpone the discussion on how the interface works to a later section, and we will be using `LinearScan` from the `deann` module in this section. `LinearScan` implements the correct interface and provides a rather slow exact nearest neighbors implementation, so it is ill-suited for practical work, but will be sufficient for demonstrating how `AnnEstimator` works.

We start by constructing the `LinearScan` object.
```
ls = deann.LinearScan('euclidean')
ls.fit(X)
```
Let us test the object by requesting the 3 nearest neighbors. The `query` function returns a 3-tuple (D,I,S) where D contains the information about the distances to the ith nearest neighbor, I contains the indices of the said neighbors, and S tells the number of samples looked at during the query.
```
q = Q[0,:]
D, I, S = ls.query(Q[0,:], 3)
print(D)
print(I)
print(S)
```
Expected output:
```
[[1.         1.73205081 2.        ]]
[[13241 40731 10472]]
[43500]
```
This shows the expected interface of the ANN objects.

We can now turn to constructing the `AnnEstimator`.
```
k = 50
m = 500
seed = 1234
ann = deann.AnnEstimator(h, 'exponential', k, m, ls, seed)
ann.fit(X)
Z, S = ann.query(Q)
print(np.all(S == n + m))
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))
```
Expected output:
```
True
0.04576981097158264
```
The number of samples looked at per query is thus equal to the number of samples looked by the ANN object (in the case of `LinearScan`, this would be n) plus the number of random samples.

Replicability can be ensured with the random seed parameter that can be reset with `reset_seed`, exactly as in the case of `RandomSampling` and `RandomSamplingPermuted` with the same precautions, that is, in the case of `AnnEstimatorPermuted`, data must be refit to enforce the repermutation of the dataset.

Finally, parameters can be reset with `reset_parameters` that admits two parameters in this case:
```
ann.reset_parameters(100, 1000)
Z, _ = ann.query(Q)
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))
```
Expected output:
```
0.029981634971266637
```

## Constructing ANN objects
The ANN object A is simply a Python object that contains a member function called `query(q,k)` that admits two parameters: a query vector `q` and the number of nearest neighbors `k`. The latter parameter is equal to the parameter k of `AnnEstimator`; the former is an individual vector that corresponds to one of the rows of the query set `Q` that is supplied to the estimator upon query. That is, it is a Numpy `ndarray` of length d.

The `query` function of A may return any of the following:
* A 1-dimensional Numpy-array of length `k` containing the *integer* indices of the approximate nearest neighbors in the dataset X. Dtype should be int32 or int64.
* A 2-tuple where the first element is a 1-dimensional Numpy array of length `k` containing the distances from the query vector to the ith nearest neighbor, and the second element is the indices as in the first case.
* A 2-tuple where the first element is the indices as in the first case and the second element is a 1-dimensional Numpy array of integers of length 1 containing information of the number of samples looked during the query.
* A 3-tuple where the first element is the distances as in the second case, the second element is the indices as in the first case, and the third element is the number of samples, as in the third case.

Note that the only *compulsory* value returned is the index vector of the nearest neighbors. However, it is recommended to also return the distances if these are efficiently available because the distances can then be directly used to compute the KDE values without having to compute again which tends to be expensive. The number of samples is only used for reporting the total number of samples used per query; if this information is not available, a -1 is reported.

If the underlying ANN implementation fails to produce k approximate nearest neighbors, the missing values can be filled as -1s; such values are ignored by the implementation.

The file `extern/brute.py` shows how the `NearestNeighbors` object from scikit-learn can be wrapped to meet the requirements of this interface. We also provide a class called `LinearScan` in the `deann` module that meets these criteria. The class implements a (rather slow) version of linear scan, so it is not suited for practical work. For practical work, we include a wrapper for FAISS under `extern/faiss.py`.


## Using FAISS

We finally show a practical example where we use FAISS as the ANN backend. In this example, we use `AnnEstimatorPermuted` and do not fix the seed.

We provide a wrapper for FAISS under `extern/faiss.py`. We also provide a base class that can be used to assist in constructing the objects; it is not necessary to subclass this class, however, it is only provided for convenience and for assisting in creating experimental pipelines.

FAISS takes two parameters: a construction-time parameter called `n_list` which describes how many clusters to construct for the data, and `n_query` which tells how many clusters to query for nearest neighbors. With our wrapper, we can work as follows.
```
from extern.faiss import FaissIVF
fivf = FaissIVF('euclidean', 1024)
fivf.fit(X)
fivf.set_query_arguments(5)

k = 150
m = 1500
annp = deann.AnnEstimatorPermuted(h, 'exponential', k, m, fivf)
annp.fit(X)
Z, S = annp.query(Q)
print(np.mean(S))
print(np.mean(np.abs(Z[idx]-mu[idx])/mu[idx]))
```
Expected output:
```
2783.9392413793103
0.024062934065316507
```
Clearly, a lot fewer samples were looked at, but the resulting error was lower.
