# DEANN - Density Estimation from Approximate Nearest Neighbors

## Introduction

This library implements the DEANN algorithm for computing Kernel Density Estimate (KDE) values as described in *arXiv link pending.* The library is written in C++ and can be compiled as a Python module. The library uses [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html) as backend for performant single-CPU computation.

The basic idea behind our estimator is that it can take an arbitrary Python object that implements a simple interface for providing approximate nearest neighbors whose contribution is computed exactly, and this estimate is completed with a random sampling estimate.

The following kernels are supported by the library:
* `'exponential'` for K_h(x,y) = exp(-||x-y||_2 / h),
* `'gaussian'` for K_h(x,y) = exp(-||x-y||_2^2 / 2 / h^2), and
* `'laplacian'` for K_h(x,y) = exp(-||x-y||_1 / h).

The following algorithms are provided in the library:
* `NaiveKde` implements the KDE summation naively from definition, using matrix multiplication as an accelerating primitive in case the kernel is Euclidean,
* `RandomSampling` implements the simple RS estimator
* `RandomSamplingPermuted` implements the RS estimator with permuted random sampling (only Euclidean kernels)
* `AnnEstimator` implements the basic DEANN estimator
* `AnnEstimatorPermuted` implements the DEANN estimator with permuted random sampling

## Building
See the instructions on a [separate page](BUILDING.md).

## Tutorial
See the tutorial on a [separate page](TUTORIAL.md).

## Reference

If you use this library in academic work, please cite the following paper:
* (arXiv link will be posted here once it's out)

## License

The project is licensed under the [MIT license](LICENSE). 
