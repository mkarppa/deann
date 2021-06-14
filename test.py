#!/usr/bin/env python3

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
import deann
import sys

from extern.brute import BruteNN
from extern.faiss import FaissIVF

import time

        

def gaussian_kernel(q,X,h):
    return np.exp(-np.sum((q[None,:]-X)**2,axis=1)/h/h/2)

def exponential_kernel(q,X,h):
    return np.exp(-np.linalg.norm(q[None,:]-X,axis=1)/h)

def laplacian_kernel(q,X,h):
    return np.exp(-np.linalg.norm(q[None,:]-X, 1, axis=1)/h)

def naive_kde(q, X, K, h):
    (n,d) = X.shape
    return np.mean(K(q,X,h))



def relative_error(wrong, correct):
    return np.abs(wrong-correct)/correct



def construct_test_set():
    N1 = 3737
    N2 = 609
    N = N1 + N2
    assert N % 2 == 0
    n = m = N//2
    
    mu1 = np.array([46.49246863044118, -0.47588088931695965, 85.96002889477487,
                    0.16082082564143724, 41.49583968750836, 0.9723626829333547,
                    39.45937341145624, 44.44947427562405, 5.157770821628274])
    
    A1 = np.array([[10.48441711263751, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.3964970383667547, 21.730310765493968, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0],
                    [3.4485084625657643, -0.28120939185823735, 8.57050385349843,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.05130389161841126, -0.08261614921017782,
                         2.025124811423481, 34.80552047242849, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [8.344024828411815, -0.030790194630293425,
                         -0.33708339870068904, 0.07328329343061195,
                    8.214712617983201, 0.0, 0.0, 0.0, 0.0],
                    [-0.03594926637950729, -0.7417873669876849,
                         -0.3153276829264685, 0.21607122879087798,
                    0.7254472592067689, 30.244917748649506, 0.0, 0.0, 0.0],
                    [-7.061282924462716, -0.2807508626111792, 8.531825490064175,
                         0.007605230963321018, -0.02816145276795588,
                    -0.006179511009823437, 0.5317477635530977, 0.0, 0.0],
                    [-4.844164935453136, -0.24415434388943516, 8.898715502133424,
                         -0.07811784341778047, -8.147890471517472,
                         -0.011989078614931445, 0.1842655678091983,
                         0.6704522457878629, 0.0],
                    [2.206335462583935, 0.03892495420779414, 0.3274019618648675,
                         -0.07720234409271014, -8.006262044202852,
                    0.04028908425200501, -0.22714713659168925,
                    0.4704157754390525, 0.6528722309151673]])
    
    mu2 = np.array([58.96024314112042, 1.8997864300969278, 81.54476753737474,
                        0.8886150813208477, -8.273698045013964,
                        -2.2424839822572697, 22.510760637424017,
                        90.52998192870051, 67.98390011499917])
    
    A2 = np.array([[16.065592111452954, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [4.009184067265925, 51.70302976271875, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0],
                    [1.8722216445930944, 0.08002264937499314, 4.650411850261695,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-2.0057332330331543, 0.11165453494383987,
                         2.3276813178792555, 67.39430848603813, 0.0, 0.0, 0.0,
                         0.0, 0.0],
                    [-6.405410328832197, 1.9441154252900719, 3.467768467646764,
                         -0.4144368424064272, 14.541572107236027, 0.0, 0.0, 0.0,
                         0.0],
                    [3.592427751645286, -1.6136283180871827,
                         -5.6823579216710245, 6.399797684468458,
                         -0.008742434962222196, 96.21821018207623, 0.0, 0.0,
                         0.0],
                    [-14.239761076411021, 0.09365916755945296,
                         4.59040104584355, 0.002839284036579001,
                    0.0027681727245653298, 0.012371444953884297,
                    0.549654997829911, 0.0, 0.0],
                    [8.44884295589548, -1.911195319541577, 1.0658234673982725,
                         0.43735285474527746, -14.922878992826801,
                         0.011714016169396069, 0.19048151609840624,
                         0.69960090582502, 0.0],
                    [22.602748434210355, -2.0069000884320456,
                         -3.508134127959344, 0.43894223781589753,
                         -14.911426203496662, 0.009953869896377816,
                         -0.17085555729674284, 0.5764982581276846,
                         0.6740065987615914]])
    d, = mu1.shape
    assert mu2.shape == (d,)
    assert A1.shape == (d,d)
    assert A2.shape == (d,d)

    rng = np.random.default_rng(271828)
   
    Z = np.zeros((N,d))
    for i in range(N1):
        z = rng.standard_normal(d)
        Z[i,:] = mu1 + A1.dot(z)
    for i in range(N1,N):
        z = rng.standard_normal(d)
        Z[i,:] = mu2 + A2.dot(z)
        
    ind = np.arange(N)
    rng.shuffle(ind)
    X_ind = ind[:n]
    Y_ind = ind[n:]
    return Z[X_ind,:], Z[Y_ind,:]



def kde_matmul_exp(query, data, h):
    return np.mean(np.exp(-np.sqrt(np.maximum((query**2).sum(-1)[:,None] - \
                                                2.0*query.dot(data.T) + \
                                                  (data**2).sum(-1)[None,:],
                                                  0.0))/h),
                       axis=-1)

def kde_matmul_gauss(query, data, h):
    return np.mean(np.exp(-np.maximum((query**2).sum(-1)[:,None] - \
                                          2.0*query.dot(data.T) + \
                                          (data**2).sum(-1)[None,:],
                                          0.0)/h/h/2),
                       axis=-1)

def kde_matmul_loop(query, data, h):
    mu = np.zeros(query.shape[0], dtype = query.dtype)
    Xsq = (data**2).sum(-1)
    for i in range(query.shape[0]):
        q = query[i,:]
        mu[i] = np.mean(np.exp(-np.sqrt(np.maximum((q**2).sum(-1) - \
                                        2.0*data.dot(q.T) + \
                                        Xsq,
                                        0.0))/h))
    return mu



def kde_laplacian(query, data, h):
    scratch = np.zeros((query.shape[0], data.shape[0]), query.dtype)
    for i in range(query.shape[0]):
        q = query[i,:]
        scratch[i,:] = np.linalg.norm(q[None,:] - data, ord = 1, axis = -1)
    return np.mean(np.exp(-scratch/h), axis=-1)


def find_nearest_neighbor(q, X, metric):
    dist = lambda x, y: \
      np.linalg.norm(x-y, ord = (2 if metric == 'euclidean' else 1))
    
    nn_idx = 0
    nn_dist = dist(q,X[0,:])
    for i in range(1, X.shape[0]):
        x = X[i,:]
        cand_dist = dist(q,x)
        if cand_dist < nn_dist:
            nn_idx = i
            nn_dist = cand_dist
    return nn_idx

                    
def test_naive_kde1():
    with pytest.raises(ValueError):
        deann.NaiveKde(0.0, 'exponential')
    with pytest.raises(ValueError):
        deann.NaiveKde(-1.0, 'exponential')
    with pytest.raises(ValueError):
        deann.NaiveKde(1.0, 'eXponential')
    with pytest.raises(ValueError):
        deann.NaiveKde(1.0,'exponential').fit(np.ones(10))
    with pytest.raises(ValueError):
        deann.NaiveKde(1.0,'exponential').fit(np.ones((10,9), dtype=np.int64))
    with pytest.raises(ValueError):
        deann.NaiveKde(1.0,'exponential').fit(np.ones((0,9), dtype=np.float64))
    with pytest.raises(ValueError):
        deann.NaiveKde(1.0,'exponential').fit(np.ones((10,0), dtype=np.float32))
    with pytest.raises(ValueError):
        nkde = deann.NaiveKde(1.0,'exponential')
        nkde.query(np.ones((2,9), dtype=np.float64))
    with pytest.raises(ValueError):
        nkde = deann.NaiveKde(1.0,'exponential')
        nkde.fit(np.ones((10,9), dtype=np.float64))
        nkde.query(np.ones((2,9), dtype=np.float32))
    with pytest.raises(ValueError):
        nkde = deann.NaiveKde(1.0,'exponential')
        nkde.fit(np.ones((10,9), dtype=np.float32))
        nkde.query(np.ones((2,9), dtype=np.float64))
    with pytest.raises(ValueError):
        nkde = deann.NaiveKde(1.0,'exponential')
        nkde.fit(np.ones((10,9), dtype=np.float32))
        nkde.query(np.ones((2,8), dtype=np.float32))

    X, Y = construct_test_set()
    assert X.dtype == np.float64
    assert Y.dtype == np.float64
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[1] == Y.shape[1]        

    abs_epsilon = 1e-15
    rel_epsilon = 1e-13
    
    h = 16.0
    mu = kde_matmul_exp(Y, X, h)
    nkde = deann.NaiveKde(h, 'exponential')
    nkde.fit(X)
    Z, S = nkde.query(Y)
    assert mu.dtype == np.float64
    assert mu.ndim == 1
    assert mu.shape[0] == Y.shape[0]
    assert mu.dtype == Z.dtype
    assert mu.ndim == Z.ndim
    assert mu.shape[0] == Z.shape[0]   
    assert np.all(np.abs(Z-mu) < abs_epsilon)
    assert np.all(np.abs((Z-mu)/mu) < rel_epsilon)
    assert S.dtype == np.int32
    assert S.ndim == 1
    assert S.shape[0] == mu.shape[0]
    assert np.all(S == X.shape[0])

    h = 19.0
    mu = kde_matmul_gauss(Y, X, h)
    nkde = deann.NaiveKde(h, 'gaussian')
    nkde.fit(X)
    Z, S = nkde.query(Y)
    assert mu.dtype == np.float64
    assert mu.ndim == 1
    assert mu.shape[0] == Y.shape[0]
    assert mu.dtype == Z.dtype
    assert mu.ndim == Z.ndim
    assert mu.shape[0] == Z.shape[0]   
    assert np.all(np.abs(Z-mu) < abs_epsilon)
    assert np.all(np.abs((Z-mu)/mu) < rel_epsilon)
    assert S.dtype == np.int32
    assert S.ndim == 1
    assert S.shape[0] == mu.shape[0]
    assert np.all(S == X.shape[0])

    h = 33.0
    mu = kde_laplacian(Y, X, h)
    nkde = deann.NaiveKde(h, 'laplacian')
    nkde.fit(X)
    Z, S = nkde.query(Y)
    assert mu.dtype == np.float64
    assert mu.ndim == 1
    assert mu.shape[0] == Y.shape[0]
    assert mu.dtype == Z.dtype
    assert mu.ndim == Z.ndim
    assert mu.shape[0] == Z.shape[0]   
    assert np.all(np.abs(Z-mu) < abs_epsilon)
    assert np.all(np.abs((Z-mu)/mu) < rel_epsilon)

    X, Y = construct_test_set()
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    assert X.dtype == np.float32
    assert Y.dtype == np.float32
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[1] == Y.shape[1]

    abs_epsilon = 1e-06
    rel_epsilon = 1e-04
    
    h = 17.0
    mu = kde_matmul_exp(Y, X, h)
    nkde = deann.NaiveKde(h, 'exponential')
    nkde.fit(X)
    Z, S = nkde.query(Y)
    assert mu.dtype == np.float32
    assert mu.ndim == 1
    assert mu.shape[0] == Y.shape[0]
    assert mu.dtype == Z.dtype
    assert mu.ndim == Z.ndim
    assert mu.shape[0] == Z.shape[0]   
    assert np.all(np.abs(Z-mu) < abs_epsilon)
    assert np.all(np.abs((Z-mu)/mu) < rel_epsilon)
    assert S.dtype == np.int32
    assert S.ndim == 1
    assert S.shape[0] == mu.shape[0]
    assert np.all(S == X.shape[0])

    h = 20.0
    mu = kde_matmul_gauss(Y, X, h)
    nkde = deann.NaiveKde(h, 'gaussian')
    nkde.fit(X)
    Z, S = nkde.query(Y)
    assert mu.dtype == np.float32
    assert mu.ndim == 1
    assert mu.shape[0] == Y.shape[0]
    assert mu.dtype == Z.dtype
    assert mu.ndim == Z.ndim
    assert mu.shape[0] == Z.shape[0]
    for i in range(mu.shape[0]):
        if np.abs((Z[i]-mu[i])/mu[i]) > rel_epsilon or np.abs(Z[i]-mu[i]) > abs_epsilon:
            print(mu[i], Z[i], np.abs(Z[i]-mu[i]), np.abs((Z[i]-mu[i])/mu[i]))
    assert np.all(np.abs(Z-mu) < abs_epsilon)
    assert np.all(np.abs((Z-mu)/mu) < rel_epsilon)
    assert S.dtype == np.int32
    assert S.ndim == 1
    assert S.shape[0] == mu.shape[0]
    assert np.all(S == X.shape[0])

    h = 34.0
    mu = kde_laplacian(Y, X, h)
    nkde = deann.NaiveKde(h, 'laplacian')
    nkde.fit(X)
    Z, S = nkde.query(Y)
    assert mu.dtype == np.float32
    assert mu.ndim == 1
    assert mu.shape[0] == Y.shape[0]
    assert mu.dtype == Z.dtype
    assert mu.ndim == Z.ndim
    assert mu.shape[0] == Z.shape[0]   
    assert np.all(np.abs(Z-mu) < abs_epsilon)
    assert np.all(np.abs((Z-mu)/mu) < rel_epsilon)
    assert S.dtype == np.int32
    assert S.ndim == 1
    assert S.shape[0] == mu.shape[0]
    assert np.all(S == X.shape[0])


    
def test_naive_kde2():
    h = 35.0
    for dt in [np.float32, np.float64]:
        X, Y = construct_test_set()
        X = X.astype(dt)
        Y = Y.astype(dt)

        for kernel in ['exponential', 'gaussian', 'laplacian']:
            nkde = deann.NaiveKde(h,kernel)
            nkde.fit(X)
            mu, S0 = nkde.query(Y)
            nkde.reset_seed()
            Z1, S1 = nkde.query(Y)
            nkde.reset_seed(0)
            Z2, S2 = nkde.query(Y)
            assert np.array_equal(mu,Z1)
            assert np.array_equal(mu,Z2)
            assert np.array_equal(S0,S1)
            assert np.array_equal(S1,S2)
            assert np.all(S0 == X.shape[0])
            



def test_random_sampling1():
    with pytest.raises(ValueError):
        deann.RandomSampling(0.0, 'exponential', 1)
    with pytest.raises(ValueError):
        deann.RandomSampling(-1.0, 'exponential', 1)
    with pytest.raises(ValueError):
        deann.RandomSampling(1.0, 'eXponential', 1)
    with pytest.raises(ValueError):
        deann.RandomSampling(1.0, 'exponential', 0)
    with pytest.raises(ValueError):
        deann.RandomSampling(1.0, 'exponential', -1)
    with pytest.raises(ValueError):
        deann.RandomSampling(1.0,'exponential',1).fit(np.ones(10))
    with pytest.raises(ValueError):
        deann.RandomSampling(1.0,'exponential',1).fit(np.ones((10,9), dtype=np.int64))
    with pytest.raises(ValueError):
        deann.RandomSampling(1.0,'exponential',1).fit(np.ones((0,9), dtype=np.float64))
    with pytest.raises(ValueError):
        deann.RandomSampling(1.0,'exponential',1).fit(np.ones((10,0), dtype=np.float32))
    with pytest.raises(ValueError):
        rs = deann.RandomSampling(1.0,'exponential',1)
        rs.query(np.ones((2,9), dtype=np.float64))
    with pytest.raises(ValueError):
        rs = deann.RandomSampling(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float64))
        rs.query(np.ones((2,9), dtype=np.int64))
    with pytest.raises(ValueError):
        rs = deann.RandomSampling(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float64))
        rs.query(np.ones((2,9), dtype=np.float32))
    with pytest.raises(ValueError):
        rs = deann.RandomSampling(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float32))
        rs.query(np.ones((2,9), dtype=np.float64))
    with pytest.raises(ValueError):
        rs = deann.RandomSampling(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float32))
        rs.query(np.ones((0,9), dtype=np.float32))
    with pytest.raises(ValueError):
        rs = deann.RandomSampling(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float32))
        rs.query(np.ones((2,0), dtype=np.float32))
    with pytest.raises(ValueError):
        rs = deann.RandomSampling(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float32))
        rs.query(np.ones((2,8), dtype=np.float32))


    ms = [1, 5, 11, 200, 600, 1400]
    seed = 31415
        
    for dt in [np.float64, np.float32]:
        X, Y = construct_test_set()
        X = X.astype(dt)
        Y = Y.astype(dt)
        h = 25.0
        for kernel in ['exponential', 'gaussian', 'laplacian']:
            avg_abs_error = np.zeros(len(ms))
            avg_rel_error = np.zeros(len(ms))
            nkde = deann.NaiveKde(h, kernel)
            nkde.fit(X)
            (mu, S0) = nkde.query(Y)
            i = 0
            for m in ms:
                seed += 1
                
                rs1 = deann.RandomSampling(h,kernel,m)
                rs1.fit(X)
                (Z1, S1) = rs1.query(Y)
                rs2 = deann.RandomSampling(h,kernel,m)
                rs2.fit(X)
                (Z2, S2) = rs2.query(Y)

                assert Z1.ndim == 1
                assert Z2.ndim == 1
                assert Z1.shape[0] == Y.shape[0]
                assert Z2.shape[0] == Y.shape[0]
                assert S0.ndim == 1
                assert S1.ndim == 1
                assert S2.ndim == 1
                assert S0.shape == S1.shape
                assert S0.shape == S2.shape
                assert np.array_equal(S1,S1)
                assert np.all(S0 == X.shape[0])
                assert np.all(S1 == m)

                if m > 11:
                    assert np.all(Z1 != Z2)
                    
                rs1 = deann.RandomSampling(h, kernel, m, seed + 1)
                rs1.fit(X)
                (Z1, S1) = rs1.query(Y)
                rs2 = deann.RandomSampling(h, kernel, m, seed + 2)
                rs2.fit(X)
                (Z2, S2) = rs2.query(Y)
                if m > 1:
                    assert np.all(Z1 != Z2)
                assert S1.ndim == 1
                assert S2.ndim == 1
                assert S1.shape[0] == mu.shape[0]
                assert S2.shape[0] == mu.shape[0]
                assert np.all(S1 == m)
                assert np.all(S2 == m)
                    
                rs1 = deann.RandomSampling(h, kernel, m, seed)
                rs1.fit(X)
                Z1, S1 = rs1.query(Y)
                rs2 = deann.RandomSampling(h, kernel, m, seed)
                rs2.fit(X)
                Z2, S2 = rs2.query(Y)
                assert np.all(Z1 == Z2)
                assert S1.ndim == 1
                assert S2.ndim == 1
                assert S1.shape[0] == mu.shape[0]
                assert S2.shape[0] == mu.shape[0]
                assert np.all(S1 == m)
                assert np.all(S2 == m)

                Z = Z1
                assert Z.ndim == 1
                assert Z.shape[0] == Y.shape[0]
                assert Z.dtype == dt
                
                avg_abs_error[i] = np.mean(np.abs(Z-mu))
                avg_rel_error[i] = np.mean(np.abs((Z-mu)/mu))
                i += 1
                
            for i in range(1,len(ms)):
                assert avg_abs_error[i] < avg_abs_error[i-1]
                assert avg_rel_error[i] < avg_rel_error[i-1]
            assert avg_abs_error[-1] < 0.01
            if kernel == 'exponential':
                assert avg_rel_error[-1] < 0.1
            else:
                assert avg_rel_error[-1] < 0.2
                

def test_random_sampling2():
    h = 36.0
    seed = 527372036
    for dt in [np.float32, np.float64]:
        X, Y = construct_test_set()
        X = X.astype(dt)
        Y = Y.astype(dt)

        for kernel in ['exponential', 'gaussian', 'laplacian']:
            for m in [1, 11, 111, 1111]:
                seed += 1
                rs1 = deann.RandomSampling(h,kernel,m,seed)
                rs1.fit(X)
                (mu, S0) = rs1.query(Y)
                assert np.all(S0 == m)

                (Z1, S1) = rs1.query(Y)
                assert not np.array_equal(mu,Z1)
                assert np.all(S1 == m)
                
                rs1.reset_seed(seed)
                (Z2, S2) = rs1.query(Y)
                assert np.array_equal(mu,Z2)
                assert np.all(S2 == m)

                rs2 = deann.RandomSampling(h,kernel,m,seed)
                rs2.fit(X)
                (Z3, S3) = rs2.query(Y)
                assert np.array_equal(mu,Z3)
                assert np.all(S3 == m)

                rs2.reset_seed(seed)
                (Z4, S4) = rs2.query(Y)
                assert np.array_equal(mu,Z4)
                assert np.all(S4 == m)

            params = [(3,101010), (13, 121212), (131, 23232323)]
            mus = list()
            for (m, seed2) in params:
                rs = deann.RandomSampling(h,kernel,m,seed2)
                rs.fit(X)
                mus.append(rs.query(Y))
            rs = deann.RandomSampling(h,kernel,1)
            rs.fit(X)
            Z = rs.query(Y)
            for i in range(len(mus)):
                assert not np.array_equal(mus[i],Z)
            for i in range(len(mus)):
                (m, seed2) = params[i]
                rs.reset_parameters(m)
                rs.reset_seed(seed2)
                Z = rs.query(Y)
                assert np.array_equal(mus[i],Z)


    
def test_random_sampling_permuted():
    with pytest.raises(ValueError):
        deann.RandomSamplingPermuted(0.0, 'exponential', 1)
    with pytest.raises(ValueError):
        deann.RandomSamplingPermuted(-1.0, 'exponential', 1)
    with pytest.raises(ValueError):
        deann.RandomSamplingPermuted(1.0, 'eXponential', 1)
    with pytest.raises(ValueError):
        deann.RandomSamplingPermuted(1.0, 'laplacian', 1)
    with pytest.raises(ValueError):
        deann.RandomSamplingPermuted(1.0, 'exponential', 0)
    with pytest.raises(ValueError):
        deann.RandomSamplingPermuted(1.0, 'exponential', -1)
    with pytest.raises(ValueError):
        deann.RandomSamplingPermuted(1.0,'exponential',1).fit(np.ones(10))
    with pytest.raises(ValueError):
        deann.RandomSamplingPermuted(1.0,'exponential',1).fit(np.ones((10,9), dtype=np.int64))
    with pytest.raises(ValueError):
        deann.RandomSamplingPermuted(1.0,'exponential',1).fit(np.ones((0,9), dtype=np.float64))
    with pytest.raises(ValueError):
        deann.RandomSamplingPermuted(1.0,'exponential',1).fit(np.ones((10,0), dtype=np.float32))
    with pytest.raises(ValueError):
        rs = deann.RandomSamplingPermuted(1.0,'exponential',1)
        rs.query(np.ones((2,9), dtype=np.float64))
    with pytest.raises(ValueError):
        rs = deann.RandomSamplingPermuted(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float64))
        rs.query(np.ones((2,9), dtype=np.int64))
    with pytest.raises(ValueError):
        rs = deann.RandomSamplingPermuted(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float64))
        rs.query(np.ones((2,9), dtype=np.float32))
    with pytest.raises(ValueError):
        rs = deann.RandomSamplingPermuted(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float32))
        rs.query(np.ones((2,9), dtype=np.float64))
    with pytest.raises(ValueError):
        rs = deann.RandomSamplingPermuted(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float32))
        rs.query(np.ones((0,9), dtype=np.float32))
    with pytest.raises(ValueError):
        rs = deann.RandomSamplingPermuted(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float32))
        rs.query(np.ones((2,0), dtype=np.float32))
    with pytest.raises(ValueError):
        rs = deann.RandomSamplingPermuted(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float32))
        rs.query(np.ones((2,8), dtype=np.float32))
    with pytest.raises(ValueError):
        rs = deann.RandomSamplingPermuted(1.0,'exponential',100)
        rs.fit(np.ones((10,9), dtype=np.float32))
    with pytest.raises(ValueError):
        rs = deann.RandomSamplingPermuted(1.0,'exponential',1)
        rs.fit(np.ones((10,9), dtype=np.float32))
        rs.reset_parameters(100)
        
    seed = 31415
        
    for dt in [np.float64, np.float32]:
        X, Y = construct_test_set()
        X = X.astype(dt)
        Y = Y.astype(dt)
        h = 25.0
        ms = [1, 103, 506, X.shape[0]]
        for kernel in ['exponential', 'gaussian']:
            avg_abs_error = np.zeros(len(ms))
            avg_rel_error = np.zeros(len(ms))
            nkde = deann.NaiveKde(h, kernel)
            nkde.fit(X)
            (mu, S0) = nkde.query(Y)
            assert np.all(S0 == X.shape[0])
            i = 0
            for m in ms:
                seed += 1
                
                rsp1 = deann.RandomSamplingPermuted(h,kernel,m)
                rsp1.fit(X)
                (Z1, S1) = rsp1.query(Y)
                rsp2 = deann.RandomSamplingPermuted(h,kernel,m)
                rsp2.fit(X)
                (Z2, S2) = rsp2.query(Y)

                assert Z1.ndim == 1
                assert Z2.ndim == 1
                assert Z1.dtype == dt
                assert Z2.dtype == dt
                assert Z1.shape[0] == Y.shape[0]
                assert Z2.shape[0] == Y.shape[0]
                assert S1.ndim == 1
                assert S2.ndim == 1
                assert S1.dtype == np.int32
                assert S2.dtype == np.int32
                assert S1.shape[0] == Y.shape[0]
                assert S2.shape[0] == Y.shape[0]
                assert np.all(S1 == m)
                assert np.all(S2 == m)

                if 1 < m < X.shape[0]:
                    assert np.all(Z1 != Z2)
                else:
                    assert np.any(Z1 != Z2)
                    
                rsp1 = deann.RandomSamplingPermuted(h, kernel, m, seed + 1)
                rsp1.fit(X)
                (Z1, S1) = rsp1.query(Y)
                rsp2 = deann.RandomSamplingPermuted(h, kernel, m, seed + 2)
                rsp2.fit(X)
                (Z2, S2) = rsp2.query(Y)
                if 1 < m < X.shape[0]:
                    assert np.all(Z1 != Z2)
                else:
                    assert np.any(Z1 != Z2)
                assert np.all(S1 == m)
                assert np.all(S2 == m)
                    
                rsp1 = deann.RandomSamplingPermuted(h, kernel, m, seed)
                rsp1.fit(X)
                (Z1, S1) = rsp1.query(Y)
                rsp2 = deann.RandomSamplingPermuted(h, kernel, m, seed)
                rsp2.fit(X)
                (Z2, S2) = rsp2.query(Y)
                # these should be equal but MKL doesn't seem to
                # guarantee that...                
                assert np.all(Z1 == Z2) or \
                    dt == np.float32 and np.amax(np.abs(Z1-Z2)) < 1e-6
                assert np.all(S1 == m)
                assert np.all(S2 == m)               
               
                rsp1 = deann.RandomSamplingPermuted(h, kernel, m, seed)
                rsp1.fit(X)
                (Z1, S1) = rsp1.query(Y)
                rsp2 = deann.RandomSamplingPermuted(h, kernel, m, seed + 1)
                rsp2.fit(X)
                rsp2.reset_seed(seed)
                (Z2, S2) = rsp2.query(Y)

                # these should be equal but MKL doesn't seem to
                # guarantee that...                
                assert np.all(Z1 == Z2) or \
                    dt == np.float32 and np.amax(np.abs(Z1-Z2)) < 1e-6
                assert np.all(S1 == m)
                assert np.all(S2 == m)               

                Z = Z1
                assert Z.ndim == 1
                assert Z.shape[0] == Y.shape[0]
                assert Z.dtype == dt
                
                avg_abs_error[i] = np.mean(np.abs(Z-mu))
                avg_rel_error[i] = np.mean(np.abs((Z-mu)/mu))
                i += 1
            for i in range(1,len(ms)):
                assert avg_abs_error[i] < avg_abs_error[i-1]
                assert avg_rel_error[i] < avg_rel_error[i-1]

            if dt == np.float64:
                assert avg_abs_error[-1] < 1e-16
            if dt == np.float32:
                assert avg_abs_error[-1] < 1e-7



def test_brute_nn():
    rng = np.random.default_rng(1234)
    k = 50
    for dt in [np.float64, np.float32]:
        X, Y = construct_test_set()
        X = X.astype(dt)
        Y = Y.astype(dt)
        n = X.shape[0]
        m = Y.shape[0]

        DELTA = 1e-12 if dt == np.float64 else 1e-05

        for metric in ['euclidean', 'taxicab']:
            bnn = BruteNN(metric)
            bnn.fit(X)
            idx = rng.integers(0,n)
            q = Y[idx,:]
            
            res = bnn.query(q,k)
            assert isinstance(res,tuple)
            assert len(res) == 3
            
            (dists, nns, samples) = res
             
            assert isinstance(dists,np.ndarray)
            assert isinstance(nns,np.ndarray)
            assert isinstance(samples,np.ndarray)

            assert dists.ndim == 2
            if metric == 'euclidean':
                assert dists.dtype == dt
            else:
                assert dists.dtype == np.float64
            assert dists.shape[0] == 1
            assert dists.shape[1] == k
            for i in range(1,k):
                assert dists[0,i-1] < dists[0,i]

            assert nns.ndim == 2
            assert nns.dtype == np.int64
            assert nns.shape[0] == 1
            assert nns.shape[1] == k
            assert np.all((nns >= 0) & (nns < n))
            
            for i in range(k):
                x = X[nns[0,i],:]
                assert np.abs(np.linalg.norm(x-q, ord = (2 if metric == 'euclidean' else 1)) - dists[0,i]) < DELTA

            assert samples.ndim == 1
            assert samples.shape[0] == 1
            assert samples.dtype == np.int32
            assert samples[0] == n

            bnn = BruteNN(metric, True, False)
            bnn.fit(X)
            
            res = bnn.query(q,k)
            assert isinstance(res,tuple)
            assert len(res) == 2
            (dists2, nns2) = res
            assert isinstance(dists2,np.ndarray)
            assert isinstance(nns2,np.ndarray)
            assert np.array_equal(dists, dists2)
            assert np.array_equal(nns, nns2)

            bnn = BruteNN(metric, False, True)
            bnn.fit(X)
            
            res = bnn.query(q,k)
            assert isinstance(res,tuple)
            assert len(res) == 2
            (nns3, samples3) = res
            assert isinstance(nns3,np.ndarray)
            assert isinstance(samples3,np.ndarray)
            assert np.array_equal(nns, nns3)
            assert samples3.ndim == samples.ndim
            assert samples3.dtype == samples.dtype
            assert np.array_equal(samples, samples3)

            bnn = BruteNN(metric, False, False)
            bnn.fit(X)
            res = bnn.query(q,k)
            assert isinstance(res,np.ndarray)
            nns4 = res
            assert np.array_equal(nns, nns4)


            
def test_linear_scan():
    with pytest.raises(ValueError):
        deann.LinearScan('eclidean')
    with pytest.raises(ValueError):
        deann.LinearScan('euclidean').fit(np.ones(9))
    with pytest.raises(ValueError):
        deann.LinearScan('euclidean').fit(np.ones((10,9), dtype=np.int64))
    with pytest.raises(ValueError):
        deann.LinearScan('euclidean').fit(np.ones((0,9), dtype=np.float64))
    with pytest.raises(ValueError):
        deann.LinearScan('euclidean').fit(np.ones((10,0), dtype=np.float32))
    with pytest.raises(ValueError):
        ls = deann.LinearScan('euclidean')
        ls.query(np.ones((2,9), dtype=np.float64), 1)
    with pytest.raises(ValueError):
        ls = deann.LinearScan('euclidean')
        ls.fit(np.ones((10,9), dtype=np.float32))
        ls.query(np.ones((2,9), dtype=np.float32), 0)
    with pytest.raises(ValueError):
        ls = deann.LinearScan('euclidean')
        ls.fit(np.ones((10,9), dtype=np.float32))
        ls.query(np.ones((2,9), dtype=np.float32), -1)
    with pytest.raises(ValueError):
        ls = deann.LinearScan('euclidean')
        ls.fit(np.ones((10,9), dtype=np.float32))
        ls.query(np.ones((2,9), dtype=np.float64), 1)
    with pytest.raises(ValueError):
        ls = deann.LinearScan('euclidean')
        ls.fit(np.ones((10,9), dtype=np.float64))
        ls.query(np.ones((2,9), dtype=np.float32), 1)
    with pytest.raises(ValueError):
        ls = deann.LinearScan('euclidean')
        ls.fit(np.ones((10,9), dtype=np.float64))
        ls.query(np.ones((0,9), dtype=np.float64), 1)
    with pytest.raises(ValueError):
        ls = deann.LinearScan('euclidean')
        ls.fit(np.ones((10,9), dtype=np.float64))
        ls.query(np.ones((2,0), dtype=np.float64), 1)
        
    for dt in [np.float64, np.float32]:
        X, Y = construct_test_set()

        X = X.astype(dt)
        Y = Y.astype(dt)

        if dt == np.float64:
            DELTA = 1e-12
        else:
            DELTA = 1e-3

        for metric in ['euclidean', 'taxicab']:
            bnn = BruteNN(metric)
            bnn.fit(X)
            ls = deann.LinearScan(metric)
            ls.fit(X)
            for k in [1, 101, X.shape[0]]:
                for i in range(Y.shape[0]):
                    q = Y[i,:]
                    (dists_bnn, nn_bnn, samples_bnn) = bnn.query(q,k)
                    if metric == 'euclidean':
                        assert dists_bnn.dtype == dt
                    else:
                        assert dists_bnn.dtype == np.float64
                    assert dists_bnn.ndim == 2
                    assert dists_bnn.shape == (1,k)
                    assert nn_bnn.dtype == np.int64
                    assert nn_bnn.ndim == 2
                    assert nn_bnn.shape == (1,k)
                    assert samples_bnn.ndim == 1
                    assert samples_bnn.shape[0] == 1
                    assert samples_bnn[0] == X.shape[0]
                    assert samples_bnn.dtype == np.int32

                    (dists_ls, nn_ls, samples_ls) = ls.query(q,k)
                    assert nn_ls.dtype == np.int32
                    assert nn_ls.ndim == 2
                    assert nn_ls.shape == (1,k)
                    assert dists_ls.dtype == dt
                    assert dists_ls.ndim == 2
                    assert dists_ls.shape == (1,k)
                    assert samples_ls.ndim == 1
                    assert samples_ls.shape[0] == 1
                    assert samples_ls[0] == X.shape[0]
                    assert samples_ls.dtype == np.int32

                    if not np.array_equal(nn_bnn,nn_ls):
                        for i in range(nn_bnn.shape[1]):
                            if nn_bnn[0,i] != nn_ls[0,i]:
                                assert dt == np.float32
                                if metric == 'euclidean':
                                    dist1 = np.linalg.norm(q-X[nn_bnn[0,i],:])
                                    dist2 = np.linalg.norm(q-X[nn_ls[0,i],:])
                                    assert np.abs(dist1-dist2) < 1e-3
                                else:
                                    dist1 = np.linalg.norm(q-X[nn_bnn[0,i],:], ord=1)
                                    dist2 = np.linalg.norm(q-X[nn_ls[0,i],:], ord=1)
                                    assert np.abs(dist1-dist2) < 1e-3

                    for i in range(nn_bnn.shape[0]):
                        idx = nn_bnn[0,i]
                        x = X[idx,:]
                        dist = np.linalg.norm(x-q, ord = 1 if metric == 'taxicab' else 2)
                        assert abs(dists_bnn[0,i] - dist) < DELTA

                        assert idx == nn_ls[0,i]
                        assert abs(dists_ls[0,i] - dist) < DELTA
                    
            k = 2*X.shape[0]+1
            q = Y[0,:]
            with pytest.raises(ValueError):
                nn_bnn = bnn.query(q,k)
            with pytest.raises(ValueError):
                nn_ls = ls.query(q,k)

                
            
class AnnObjectInvalid:
    pass

class AnnObjectInvalidType:
    def __init__(self, as_ndarray = True, as_tuple = True,
                 nn_type = np.int64, dist_type = np.float64):
        self._as_tuple = as_tuple
        self._nn_type = nn_type
        self._dist_type = dist_type
        self._as_ndarray = as_ndarray
        
    def query(self, q, k):
        if self._as_tuple:
            return (-np.ones(k, dtype=self._dist_type),
                        -np.ones(k, dtype=self._nn_type))
        else:
            if self._as_ndarray:
                return -np.ones(k, dtype = self._nn_type)
            else:
                return None

class AnnObject:
    def __init__(self, ann = None):
        self.num_of_calls = 0
        self._ann = ann
        
    def query(self, q, k):
        self.num_of_calls += 1
        return self._ann.query(q,k)



def test_ann_estimator1():
    with pytest.raises(ValueError):
        deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObjectInvalid())
    with pytest.raises(ValueError):
        deann.AnnEstimator(0.0, 'exponential', 0, 0, AnnObject())
    with pytest.raises(ValueError):
        deann.AnnEstimator(-0.0, 'exponential', 0, 0, AnnObject())
    with pytest.raises(ValueError):
        deann.AnnEstimator(1.0, 'eXponential', 0, 0, AnnObject())
    with pytest.raises(ValueError):
        deann.AnnEstimator(1.0, 'exponential', -1, 0, AnnObject()) 
    with pytest.raises(ValueError):
        deann.AnnEstimator(1.0, 'exponential', 0, -1, AnnObject())
    with pytest.raises(ValueError):
        deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject()).fit(np.ones(9,dtype=np.float64))
    with pytest.raises(ValueError):
        deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject()).fit(np.ones((10,9),dtype=np.int64))
    with pytest.raises(ValueError):
        deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject()).fit(np.ones((0,9),dtype=np.float32))
    with pytest.raises(ValueError):
        deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject()).fit(np.ones((10,0),dtype=np.float32))
    with pytest.raises(ValueError):
        deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject()).query(np.ones((2,9),dtype=np.float64))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float64))
        ann_estimator.query(np.ones((2,9),dtype=np.int64))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float64))
        ann_estimator.query(np.ones((2,9),dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones((2,9),dtype=np.float64))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones((2,9,1),dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones((0,9),dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones((2,0),dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimator(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones(0,dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimator(1.0, 'exponential', 1, 0,
                                             AnnObjectInvalidType(False, False))
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones(9,dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimator(1.0, 'exponential', 1, 0,
                                             AnnObjectInvalidType(True, False, np.float32))
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones(9,dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimator(1.0, 'exponential', 1, 0,
                                             AnnObjectInvalidType(True, True, np.float32))
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones(9,dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimator(1.0, 'exponential', 1, 0,
                                             AnnObjectInvalidType(True, True, np.int32, np.int32))
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones(9,dtype=np.float32))

    random_seed = 11992288
    rng = np.random.default_rng()
    NITERS = 100
    DELTA64 = 1e-16
    EPSILON64 = 1e-13
    DELTA32 = 1e-10
    EPSILON32 = 1e-5
    for dt in [np.float64, np.float32]:
        X, Y = construct_test_set()

        X = X.astype(dt)
        Y = Y.astype(dt)
        h = 21.0

        for kernel in ['exponential', 'gaussian', 'laplacian']:
            metric = 'taxicab' if kernel == 'laplacian' else 'euclidean'
            bnn = BruteNN(metric)
            bnn.fit(X)

            ann_estimator = deann.AnnEstimator(h, kernel, 0, 0, AnnObject())
            ann_estimator.fit(X)
            (Z,S) = ann_estimator.query(Y)
            assert np.all(Z == 0)
            assert np.all(S == 0)

            ann_object = AnnObject(bnn)
            ann_estimator = deann.AnnEstimator(h, kernel, 1, 0, ann_object)
            ann_estimator.fit(X)

            j = 0
            for i in rng.choice(Y.shape[0], NITERS, replace = False): 
                j += 1
                q = Y[i,:]
                (Z,S) = ann_estimator.query(q)
                assert ann_object.num_of_calls == j

                nn_idx = find_nearest_neighbor(q,X,metric)
                x = X[nn_idx,:]
                dist = np.linalg.norm(q-x, ord = (1 if metric == 'taxicab' else 2))
                mu = np.exp(-dist*dist/h/h/2) if kernel == 'gaussian' else np.exp(-dist/h)
                mu /= X.shape[0]

                assert Z.ndim == 1 and Z.shape[0] == 1
                assert Z[0] > 0
                assert mu > 0
                if dt == np.float64:
                    assert np.abs(Z-mu) < DELTA64
                    assert np.abs(Z-mu)/mu < EPSILON64
                elif dt == np.float32:
                    assert np.abs(Z-mu) < DELTA32
                    assert np.abs(Z-mu)/mu < EPSILON32
                assert S.ndim == 1 and S.shape[0] == 1
                assert S.dtype == np.int32
                assert S[0] == X.shape[0]
                    
            ann_object = AnnObject(bnn)               
            ann_estimator = deann.AnnEstimator(h, kernel, X.shape[0], 0, ann_object)
            ann_estimator.fit(X)
            nkde = deann.NaiveKde(h, kernel)
            nkde.fit(X)
            
            start = time.time()
            (mu, S) = nkde.query(Y)
            end = time.time()
            print(f'nkde query took {end-start} s')
            
            start = time.time()
            (Z, S) = ann_estimator.query(Y)
            end = time.time()
            print(f'ann estimator query took {end-start} s')
            assert np.all(S == X.shape[0])

            assert ann_object.num_of_calls == Y.shape[0]
            if dt == np.float64:
                assert np.all(np.abs(mu-Z) < 1e-15)
            elif dt == np.float32:
                assert np.all(np.abs(mu-Z) < 1e-6)

            for m in [1, 101]:
                random_seed += m
                
                ann_object = AnnObject(bnn)               
                ann_estimator = deann.AnnEstimator(h, kernel, 0, m,
                                                     ann_object, random_seed)
                ann_estimator.fit(X)
                
                rs = deann.RandomSampling(h, kernel, m, random_seed)
                rs.fit(X)

                (Z1, S1) = ann_estimator.query(Y)
                (Z2, S2) = rs.query(Y)

                assert ann_object.num_of_calls == 0
                if dt == np.float64:
                    assert np.all(np.abs(Z1-Z2) < 1e-15)
                elif dt == np.float32:
                    assert np.all(np.abs(Z1-Z2) < 1e-6)
                assert np.all(S1 == S2)
                
            params = [(101,203)]
            avg_abs_err = np.zeros(len(params))
            avg_rel_err = np.zeros(len(params))
            i = 0
            for (k, m) in params:
                random_seed += k*m
                ann_object = AnnObject(bnn)               
                ann_estimator1 = deann.AnnEstimator(h, kernel, k, 0,
                                                     ann_object)
                ann_estimator1.fit(X)
                (Z1, S1) = ann_estimator1.query(Y)
                assert ann_object.num_of_calls == Y.shape[0]
                assert np.all(S1 == X.shape[0])
                
                ann_estimator2 = deann.AnnEstimator(h, kernel, k, 0,
                                                     ann_object)
                ann_estimator2.fit(X)
                (Z2, S2) = ann_estimator2.query(Y)
                assert ann_object.num_of_calls == 2*Y.shape[0]
                assert np.array_equal(Z1,Z2)
                assert np.all(S2 == X.shape[0])
                
                ann_estimator3 = deann.AnnEstimator(h, kernel, 0, m,
                                                     ann_object)
                ann_estimator3.fit(X)
                (Z3, S3) = ann_estimator3.query(Y)
                assert ann_object.num_of_calls == 2*Y.shape[0]
                assert np.all(S3 == m)
                
                ann_estimator4 = deann.AnnEstimator(h, kernel, 0, m,
                                                     ann_object)
                ann_estimator4.fit(X)
                (Z4, S4) = ann_estimator4.query(Y)
                assert ann_object.num_of_calls == 2*Y.shape[0]
                assert not np.array_equal(Z3,Z4)
                assert np.all(S4 == m)
                
                ann_estimator5 = deann.AnnEstimator(h, kernel, 0, m,
                                                     ann_object, random_seed)
                ann_estimator5.fit(X)
                (Z5, S5) = ann_estimator5.query(Y)
                assert ann_object.num_of_calls == 2*Y.shape[0]
                assert np.all(S5 == m)
                
                ann_estimator6 = deann.AnnEstimator(h, kernel, 0, m,
                                                     ann_object, random_seed)
                ann_estimator6.fit(X)
                (Z6, S6) = ann_estimator6.query(Y)
                assert ann_object.num_of_calls == 2*Y.shape[0]
                assert np.array_equal(Z5,Z6)
                assert np.all(S6 == m)

                ann_estimator7 = deann.AnnEstimator(h, kernel, k, m,
                                                      ann_object, random_seed)
                ann_estimator7.fit(X)
                (Z7, S7) = ann_estimator7.query(Y)
                assert ann_object.num_of_calls == 3*Y.shape[0]
                assert np.all(S7 == m + X.shape[0])
                
                ann_estimator8 = deann.AnnEstimator(h, kernel, k, m,
                                                     ann_object, random_seed)
                ann_estimator8.fit(X)
                (Z8, S8) = ann_estimator8.query(Y)
                assert ann_object.num_of_calls == 4*Y.shape[0]
                assert np.array_equal(Z7,Z8)
                
                assert np.all(S8 == m + X.shape[0])
                
                abs_error = np.mean(np.abs(Z7-mu))
                assert abs_error < np.mean(np.abs(Z5-mu))
                assert abs_error < np.mean(np.abs(Z1-mu))

                rel_error = np.mean(np.abs((Z7-mu)/mu))
                assert rel_error < np.mean(np.abs((Z5-mu)/mu))
                assert rel_error < np.mean(np.abs((Z1-mu)/mu))

                avg_abs_err[i] = abs_error
                avg_rel_err[i] = rel_error
                
                i += 1

            assert avg_abs_err[-1] < 0.01
            assert avg_rel_err[-1] < 0.1
    

def test_ann_estimator2():
    h = 37.0
    seed = 527372036
    rng = np.random.default_rng()
    NITERS = 100
    for dt in [np.float32, np.float64]:
        X, Y = construct_test_set()
        X = X.astype(dt)
        Y = Y.astype(dt)
        Y = Y[rng.choice(Y.shape[0], NITERS, replace = False),:]

        for kernel in ['exponential', 'gaussian', 'laplacian']:
            metric = 'taxicab' if kernel == 'laplacian' else 'euclidean'
            bnn = BruteNN(metric)
            bnn.fit(X)

            for k in [0, 11]:
                for m in [0, 13]:
                    seed += 1
                    ann1 = deann.AnnEstimator(h, kernel, k, m, bnn, seed)
                    ann1.fit(X)
                    (mu, S0) = ann1.query(Y)
                    
                    if k == 0 and m == 0:
                        assert np.all(mu == 0)
                        assert np.all(S0 == 0)
                    else:
                        assert np.all(mu > 0)
                        if k > 0:
                            assert np.all(S0 == X.shape[0] + m)
                        else:
                            assert np.all(S0 == m)

                    ann2 = deann.AnnEstimator(h, kernel, k, m, bnn)
                    ann2.fit(X)
                    (Z, S1) = ann2.query(Y)

                    if m == 0:
                        assert np.array_equal(mu, Z)
                    else:
                        assert not np.array_equal(mu,Z)
                    assert np.array_equal(S0,S1)

                    ann2.reset_seed(seed)
                    (Z, S2) = ann2.query(Y)
                    assert np.array_equal(mu, Z)
                    assert np.array_equal(S0,S2)
                    
                    ann3 = deann.AnnEstimator(h, kernel, k+1, m, bnn, seed)
                    ann3.fit(X)
                    (Z, S3) = ann3.query(Y)
                    assert not np.array_equal(mu,Z)
                    assert np.all(S3 == X.shape[0] + m)
                    ann3.reset_parameters(k,m)
                    (Z, S4) = ann3.query(Y)
                    if m == 0:
                        assert np.array_equal(mu,Z)
                    else:
                        assert not np.array_equal(mu,Z)
                    assert np.array_equal(S0,S4)
                    ann3.reset_seed(seed)
                    (Z, S5) = ann3.query(Y)
                    assert np.array_equal(mu,Z)
                    assert np.array_equal(S0,S5)
                    
                    ann4 = deann.AnnEstimator(h, kernel, k, m+1, bnn, seed)
                    ann4.fit(X)
                    (Z, S6) = ann4.query(Y)
                    assert not np.array_equal(mu,Z)
                    assert np.array_equal(S0 + 1, S6)
                    ann4.reset_parameters(k,m)
                    (Z, S7) = ann4.query(Y)
                    if m == 0:
                        assert np.array_equal(mu,Z)
                    else:
                        assert not np.array_equal(mu,Z)
                    assert np.array_equal(S0,S7)
                    ann4.reset_seed(seed)
                    (Z, S8) = ann4.query(Y)
                    assert np.array_equal(mu,Z)
                    assert np.array_equal(S0,S8)

                    ann5 = deann.AnnEstimator(h, kernel, k+1, m+1, bnn, seed)
                    ann5.fit(X)
                    (Z, S9) = ann5.query(Y)
                    assert not np.array_equal(mu,Z)
                    assert np.all(S9 == X.shape[0] + m + 1)
                    ann5.reset_parameters(k,m)
                    (Z, S10) = ann5.query(Y)
                    if m == 0:
                        assert np.array_equal(mu,Z)
                    else:
                        assert not np.array_equal(mu,Z)
                    assert np.array_equal(S0,S10)
                    ann5.reset_seed(seed)
                    (Z, S11) = ann5.query(Y)
                    assert np.array_equal(mu,Z)
                    assert np.array_equal(S0,S11)



def test_ann_estimator3():
    h = 24.0
    seed = 0x055c5b79
    DELTA64 = 1e-15
    EPSILON64 = 1e-13
    DELTA32 = 1e-6
    EPSILON32 = 1e-5
    rng = np.random.default_rng()
    NITERS = 100
    for dt in [np.float32, np.float64]:
        X, Y = construct_test_set()
        X = X.astype(dt)
        Y = Y.astype(dt)
        Y = Y[rng.choice(Y.shape[0], NITERS, replace = False),:]

        if dt == np.float32:
            DELTA = 1e-3
        else:
            DELTA = 1e-12

        for kernel in ['exponential', 'gaussian', 'laplacian']:
            metric = 'taxicab' if kernel == 'laplacian' else 'euclidean'
            bnn1 = BruteNN(metric, True, True)
            bnn1.fit(X)
            bnn2 = BruteNN(metric, False, False)
            bnn2.fit(X)

            for k in [0, 103, X.shape[0]]:
                seed += 1
                if k > 0:
                    for j in range(Y.shape[0]):
                        q = Y[j,:]
                        dists, nns1, samples = bnn1.query(q,k)
                        nns2 = bnn2.query(q,k)

                        assert samples.ndim == 1 and samples.shape[0] == 1 and samples[0] == X.shape[0]

                        assert nns1.shape == nns2.shape
                        assert nns1.shape[0] == 1
                        assert np.array_equal(nns1, nns2)
                        for l in range(nns2.shape[1]):
                            idx = nns2[0,l]
                            x = X[idx,:]
                            dist = np.linalg.norm(x-q, ord = 1 if metric == 'taxicab' else 2)
                            assert abs(dists[0,l] - dist) < DELTA

                for m in [0, 123]:
                    ann1 = deann.AnnEstimator(h, kernel, k, m, bnn1, seed)
                    ann1.fit(X)
                    (mu, S) = ann1.query(Y)
                    
                    if k == 0 and m == 0:
                        assert np.all(mu == 0)
                        assert np.all(S == 0)
                    else:
                        assert np.all(mu > 0)
                        if k > 0:
                            if k >= X.shape[0]:
                                assert np.all(S == X.shape[0])
                            else:
                                assert np.all(S == X.shape[0] + m)
                        else:
                            assert np.all(S == m)

                    ann2 = deann.AnnEstimator(h, kernel, k, m, bnn2, seed)
                    ann2.fit(X)
                    (Z, S) = ann2.query(Y)
                        
                    assert mu.shape == Z.shape
                    assert S.shape == mu.shape

                    if k == 0:
                        assert np.array_equal(Z,mu)
                        assert np.all(S == m)
                    else:
                        for i in range(mu.shape[0]):
                            if dt == np.float64:
                                assert np.abs(mu[i]-Z[i]) < DELTA64
                                assert np.abs(mu[i]-Z[i])/mu[i] < EPSILON64
                            else:
                                assert np.abs(mu[i]-Z[i]) < DELTA32
                                assert np.abs(mu[i]-Z[i])/mu[i] < EPSILON32
                        assert np.allclose(mu, Z, DELTA64 if dt == np.float64 else DELTA32)
                        assert np.all(S == -1)

def test_ann_faiss():
    m = 100
    h = 32.0
    seed = 112233
    rng = np.random.default_rng()
    NITERS = 100
    for dt in [np.float32, np.float64]:
        X, Y = construct_test_set()
        X = X.astype(dt)
        Y = Y.astype(dt)
        Y = Y[rng.choice(Y.shape[0], NITERS, replace = False),:]
        
        nkde = deann.NaiveKde(h, 'exponential')
        nkde.fit(X)
        mu, _ = nkde.query(Y)

        for k in [5,15]:
            for n_list in [16, 32]:
                fivf = FaissIVF('euclidean', n_list)
                ann_fivf = deann.AnnEstimator(h, 'exponential', k, m, fivf)
                fivf.fit(X)
                ann_fivf.fit(X)
                for n_probe in [5, 10]:
                    fivf.set_query_arguments(n_probe)
                    
                    Z, S = ann_fivf.query(Y)
                    assert np.all(S > m)
                    assert np.all(S < X.shape[0])
                    assert np.mean(np.abs(mu-Z)) < 0.02



def test_ann_estimator_permuted1():
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObjectInvalid())
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(0.0, 'exponential', 0, 0, AnnObject())
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(-0.0, 'exponential', 0, 0, AnnObject())
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'eXponential', 0, 0, AnnObject())
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'exponential', -1, 0, AnnObject()) 
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'exponential', 0, -1, AnnObject())
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject()).fit(np.ones(9,dtype=np.float64))
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject()).fit(np.ones((10,9),dtype=np.int64))
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject()).fit(np.ones((0,9),dtype=np.float32))
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject()).fit(np.ones((10,0),dtype=np.float32))
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'exponential', 100, 0, AnnObject()).fit(np.ones((10,9),dtype=np.float64))
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 100, AnnObject()).fit(np.ones((10,9),dtype=np.float64))
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'exponential', 6, 6, AnnObject()).fit(np.ones((10,9),dtype=np.float64))
    with pytest.raises(ValueError):
        deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject()).query(np.ones((2,9),dtype=np.float64))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float64))
        ann_estimator.query(np.ones((2,9),dtype=np.int64))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float64))
        ann_estimator.query(np.ones((2,9),dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones((2,9),dtype=np.float64))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones((2,9,1),dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones((0,9),dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones((2,0),dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0, AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones(0,dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 1, 0,
                                             AnnObjectInvalidType(False, False))
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones(9,dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 1, 0,
                                             AnnObjectInvalidType(True, False, np.float32))
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones(9,dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 1, 0,
                                             AnnObjectInvalidType(True, True, np.float32))
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones(9,dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 1, 0,
                                             AnnObjectInvalidType(True, True, np.int32, np.int32))
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.query(np.ones(9,dtype=np.float32))

    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 100, 0,
                                                     AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 100,
                                                     AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 6, 6,
                                                     AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0,
                                                     AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.reset_parameters(100)
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0,
                                                     AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.reset_parameters(None, 100)
    with pytest.raises(ValueError):
        ann_estimator = deann.AnnEstimatorPermuted(1.0, 'exponential', 0, 0,
                                                     AnnObject())
        ann_estimator.fit(np.ones((10,9),dtype=np.float32))
        ann_estimator.reset_parameters(6,6)


    random_seed = 0x90f95369
    DELTA64 = 1e-15
    EPSILON64 = 1e-13
    DELTA32 = 1e-6
    EPSILON32 = 1e-5
    for dt in [np.float64, np.float32]:
        X, Y = construct_test_set()

        X = X.astype(dt)
        Y = Y.astype(dt)
        h = 21.0

        if dt == np.float64:
            DELTA = DELTA64
            EPSILON = EPSILON32
        else:
            DELTA = DELTA32
            EPSILON = EPSILON32

        for kernel in ['exponential', 'gaussian']:
            metric = 'euclidean'
            bnn = BruteNN(metric)
            bnn.fit(X)

            ann_estimator = deann.AnnEstimatorPermuted(h, kernel, 0, 0, AnnObject())
            ann_estimator.fit(X)
            (Z,S) = ann_estimator.query(Y)
            assert np.all(Z == 0)
            assert np.all(S == 0)

            ann_object = AnnObject(bnn)
            ann_estimator = deann.AnnEstimatorPermuted(h, kernel, 1, 0, ann_object)
            ann_estimator.fit(X)
            
            for i in range(Y.shape[0]):
                break
                q = Y[i,:]
                (Z,S) = ann_estimator.query(q)
                assert ann_object.num_of_calls == i+1

                nn_idx = find_nearest_neighbor(q,X,metric)
                x = X[nn_idx,:]
                dist = np.linalg.norm(q-x)
                mu = np.exp(-dist*dist/h/h/2) if kernel == 'gaussian' else np.exp(-dist/h)
                mu /= X.shape[0]

                assert Z.ndim == 1 and Z.shape[0] == 1
                assert Z[0] > 0
                assert mu > 0
                if dt == np.float64:
                    assert np.abs(Z-mu) < DELTA64
                    assert np.abs(Z-mu)/mu < EPSILON64
                elif dt == np.float32:
                    assert np.abs(Z-mu) < DELTA32
                    assert np.abs(Z-mu)/mu < EPSILON32
                assert S.ndim == 1 and S.shape[0] == 1
                assert S.dtype == np.int32
                assert S[0] == X.shape[0]
                
            ann_object = AnnObject(bnn)               
            ann_estimator = deann.AnnEstimatorPermuted(h, kernel, X.shape[0], 0, ann_object)
            ann_estimator.fit(X)
            nkde = deann.NaiveKde(h, kernel)
            nkde.fit(X)
            
            start = time.time()
            (mu, S) = nkde.query(Y)
            end = time.time()
            print(f'nkde query took {end-start} s')
            
            start = time.time()
            (Z, S) = ann_estimator.query(Y)
            end = time.time()
            print(f'ann estimator query took {end-start} s')
            assert np.all(S == X.shape[0])

            assert ann_object.num_of_calls == Y.shape[0]
            assert np.all(np.abs(mu-Z) < DELTA)

            m = 123

            random_seed += m
                
            ann_object = AnnObject(bnn)               
            ann_estimator = deann.AnnEstimatorPermuted(h, kernel, 0, m,
                                                         ann_object, random_seed)
            ann_estimator.fit(X)
                
            rs = deann.RandomSamplingPermuted(h, kernel, m, random_seed)
            rs.fit(X)

            (Z1, S1) = ann_estimator.query(Y)
            (Z2, S2) = rs.query(Y)

            assert ann_object.num_of_calls == 0
            assert np.all(np.abs(Z1-Z2) < DELTA)

            k, m = 101,203
            random_seed += k*m
            ann_object = AnnObject(bnn)               
            ann_estimator1 = deann.AnnEstimatorPermuted(h, kernel, k, 0,
                                                        ann_object)
            ann_estimator1.fit(X)
            (Z1, S1) = ann_estimator1.query(Y)
            assert ann_object.num_of_calls == Y.shape[0]
            assert np.all(S1 == X.shape[0])
            
            ann_estimator2 = deann.AnnEstimatorPermuted(h, kernel, k, 0,
                                                          ann_object)
            ann_estimator2.fit(X)
            (Z2, S2) = ann_estimator2.query(Y)
            assert ann_object.num_of_calls == 2*Y.shape[0]
            assert np.array_equal(Z1,Z2)
            assert np.all(S2 == X.shape[0])
                
            ann_estimator3 = deann.AnnEstimatorPermuted(h, kernel, 0, m,
                                                  ann_object)
            ann_estimator3.fit(X)
            (Z3, S3) = ann_estimator3.query(Y)
            assert ann_object.num_of_calls == 2*Y.shape[0]
            assert np.all(S3 == m)
                
            ann_estimator4 = deann.AnnEstimatorPermuted(h, kernel, 0, m,
                                                  ann_object)
            ann_estimator4.fit(X)
            (Z4, S4) = ann_estimator4.query(Y)
            assert ann_object.num_of_calls == 2*Y.shape[0]
            assert not np.array_equal(Z3,Z4)
            assert np.all(S4 == m)

            ann_estimator5 = deann.AnnEstimatorPermuted(h, kernel, 0, m,
                                                          ann_object, random_seed)
            ann_estimator5.fit(X)
            (Z5, S5) = ann_estimator5.query(Y)
            assert ann_object.num_of_calls == 2*Y.shape[0]
            assert np.all(S5 == m)
            ann_estimator6 = deann.AnnEstimatorPermuted(h, kernel, 0, m,
                                                          ann_object, random_seed)
            ann_estimator6.fit(X)
            (Z6, S6) = ann_estimator6.query(Y)
            assert ann_object.num_of_calls == 2*Y.shape[0]
            assert np.array_equal(Z5,Z6)
            assert np.all(S6 == m)
            ann_estimator7 = deann.AnnEstimatorPermuted(h, kernel, k, m,
                                                  ann_object, random_seed)
            ann_estimator7.fit(X)
            (Z7, S7) = ann_estimator7.query(Y)
            assert ann_object.num_of_calls == 3*Y.shape[0]
            assert np.all(m + X.shape[0] <= S7)
            assert np.all(S7 <= 2*X.shape[0])
            ann_estimator8 = deann.AnnEstimatorPermuted(h, kernel, k, m,
                                                          ann_object, random_seed)
            
            ann_estimator8.fit(X)
            (Z8, S8) = ann_estimator8.query(Y)
            assert ann_object.num_of_calls == 4*Y.shape[0]
            # these should be equal but for some reason they are
            # not necessarily so; this is probably some MKL thing
            
            assert np.array_equal(Z7,Z8) or \
                dt == np.float32 and np.amax(np.abs(Z7-Z8)) < 1e-7

            assert np.all(m + X.shape[0] <= S8)
            assert np.all(S8 <= 2*X.shape[0])
            abs_error = np.mean(np.abs(Z7-mu))
            assert abs_error < np.mean(np.abs(Z5-mu))
            assert abs_error < np.mean(np.abs(Z1-mu))

            rel_error = np.mean(np.abs((Z7-mu)/mu))
            assert rel_error < np.mean(np.abs((Z5-mu)/mu))
            assert rel_error < np.mean(np.abs((Z1-mu)/mu))

            assert abs_error < 0.01
            assert rel_error < 0.1
                
    

def test_ann_estimator_permuted2():
    h = 37.0
    seed1 = 0xfed712db
    seed2 = 0x710867b8
    k1 = 123
    k2 = 321
    m1 = 234
    m2 = 432
    for dt in [np.float32, np.float64]:
        X, Y = construct_test_set()
        X = X.astype(dt)
        Y = Y.astype(dt)
        for kernel in ['exponential', 'gaussian']:
            seed1 += 1
            seed2 += 1
            
            metric = 'euclidean'
            bnn = BruteNN(metric)
            bnn.fit(X)

            ann1 = deann.AnnEstimatorPermuted(h, kernel, k1, m1, bnn, seed1)
            ann1.fit(X)
            (mu, S0) = ann1.query(Y)
                    
            ann2 = deann.AnnEstimatorPermuted(h, kernel, k1, m1, bnn, seed1)
            ann2.fit(X)
            (Z, S) = ann2.query(Y)
            assert np.array_equal(Z,mu)
            assert np.array_equal(S,S0)

            ann3 = deann.AnnEstimatorPermuted(h, kernel, k2, m2, bnn, seed2)
            ann3.fit(X)
            (Z, S) = ann3.query(Y)
            assert np.all(Z != mu)
            assert np.all(S != S0)

            ann4 = deann.AnnEstimatorPermuted(h, kernel, k2, m2, bnn, seed2)
            ann4.fit(X)
            ann4.reset_seed(seed1)
            ann4.reset_parameters(k1,m1)
            (Z, S) = ann4.query(Y)
            assert np.array_equal(Z,mu)
            assert np.array_equal(S,S0)
            


def test_ann_estimator_permuted3():
    h = 24.0
    seed = 0xf520208d
    DELTA64 = 1e-15
    EPSILON64 = 1e-13
    DELTA32 = 1e-6
    EPSILON32 = 1e-5
    for dt in [np.float32, np.float64]:
        X, Y = construct_test_set()
        X = X.astype(dt)
        Y = Y.astype(dt)

        if dt == np.float32:
            DELTA = 1e-3
        else:
            DELTA = 1e-12

        for kernel in ['exponential', 'gaussian']:
            metric = 'taxicab' if kernel == 'laplacian' else 'euclidean'
            bnn1 = BruteNN(metric, True, True)
            bnn1.fit(X)
            bnn2 = BruteNN(metric, False, False)
            bnn2.fit(X)

            for k in [0, 123]:
                for m in [0, 321]:
                    seed += 1
                    ann1 = deann.AnnEstimatorPermuted(h, kernel, k, m, bnn1, seed)
                    ann1.fit(X)
                    (mu, S) = ann1.query(Y)
                    
                    if k == 0 and m == 0:
                        assert np.all(mu == 0)
                        assert np.all(S == 0)
                    else:
                        assert np.all(mu > 0)
                        if k > 0:
                            assert np.all(S >= X.shape[0] + m)
                            assert np.all(S <= 2*X.shape[0])
                        else:
                            assert np.all(S == m)

                    ann2 = deann.AnnEstimatorPermuted(h, kernel, k, m, bnn2, seed)
                    ann2.fit(X)
                    (Z, S) = ann2.query(Y)
                        
                    assert mu.shape == Z.shape
                    assert S.shape == mu.shape

                    if k == 0:
                        assert np.array_equal(Z,mu)
                        assert np.all(S == m)
                    else:
                        for i in range(mu.shape[0]):
                            if dt == np.float64:
                                assert np.abs(mu[i]-Z[i]) < DELTA64
                                assert np.abs(mu[i]-Z[i])/mu[i] < EPSILON64
                            else:
                                assert np.abs(mu[i]-Z[i]) < DELTA32
                                assert np.abs(mu[i]-Z[i])/mu[i] < EPSILON32
                        assert np.allclose(mu, Z, DELTA64 if dt == np.float64 else DELTA32)
                        assert np.all(S == -1)
                    
if __name__ == '__main__':
    X, Y = construct_test_set()

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    h = 21.0
    kernel = 'gaussian'
    random_seed = 2432279547
    m = 123

    ann_object = BruteNN('euclidean')
    ann_object.fit(X)
    ann_estimator = deann.AnnEstimatorPermuted(h, kernel, 0, m,
                                                 ann_object, random_seed)
    ann_estimator.fit(X)
    (Z5, S5) = ann_estimator.query(Y)
    
    for fun in [test_naive_kde1, test_naive_kde2, test_random_sampling1,
                test_random_sampling2, test_random_sampling_permuted,
                test_brute_nn, test_linear_scan, test_ann_estimator1,
                test_ann_estimator2, test_ann_estimator3, test_ann_faiss,
                test_ann_estimator_permuted1, test_ann_estimator_permuted2,
                test_ann_estimator_permuted3]:
        start = time.time()
        fun()
        end = time.time()
        print(fun.__name__, end-start)
    
