#ifndef DEANN_ARRAY
#define DEANN_ARRAY

#include "Metric.hpp"
#include <mkl.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <string>
#include <vector>

/**
 * @file 
 * This file contains a general array operation interface. The
 * operations are templatized, but work on C-like simple
 * arrays. Sometimes the operations are provided only for float and
 * double types, but some functions work with arbitrary numerical
 * types.
 *
 * In general, the order of operands is
 * - one or more array shape parameters (int),
 * - zero or more input arrays (const T*)
 * - zero or more output arrays (T*)
 * - zero or more auxiliary arrays of implied size (T*)
 * 
 * For one-dimensional arrays, \f$x[i]\f$ denotes the \f$i\f$th
 * element. For two-dimensional arrays, \f$X[i,:]\f$ and \f$X[:,j]\f$
 * denote the \f$i\f$th column and the \f$j\f$th row, respectively.
 *
 * All arrays are assumed to have row-major data layout. For linear
 * algebra purposes, untransposed vectors are considered column
 * vectors. Some functions are only thin wrappers around BLAS operations.
 */

namespace deann {
  /**
   * The namespace for array subroutines.
   */
  namespace array {
    /**
     * Computes the operation \f$x\gets \max(0,x)\f$.
     * @param d The length of x.
     * @param x Array to modify in-place.
     * @tparam T Data type.
     */
    template<typename T>
    void maxZero(int d, T* x) {
      T y;
      for (int i = 0; i < d; ++i) {
	y = x[i];
	x[i] = y > 0 ? y : 0;
      }
    }


    
    /**
     * Returns \f$||x||_2^2\f$.
     * @param d The length of x.
     * @param x Input array.
     * @return \f$||x||_2^2\f$
     * @tparam T Data type.
     */
    template<typename T>
    T sqNorm(int d, const T* x);

    template<>
    inline float sqNorm(int d, const float* x) {
      return cblas_sdot(d, x, 1, x, 1);
    }

    template<>
    inline double sqNorm(int d, const double* x) {
      return cblas_ddot(d, x, 1, x, 1);
    }



    /**
     * Computes the rowwise squared Euclidean norm of the input matrix, that is,
     * \f$y[i]\gets ||X[i,:]||_2^2\f$.
     * @param n The number of rows in X.
     * @param d The number of columns in X.
     * @param X Input array.
     * @param y Output array.
     * @tparam T Data type.
     */
    template<typename T>
    void sqNorm(int n, int d, const T* __restrict X, T* __restrict y) {
      for (int i = 0; i < n; ++i)
	y[i] = sqNorm(d, X + i*d);
    }



    /**
     * Computes the operation \f$x[i]\gets \sqrt{x[i]}\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x.
     * @param x Array to modify in-place.
     * @tparam T Data type.
     */
    template<typename T>
    void sqrt(int d, T* x);

    template<>
    void sqrt(int d, float* x) {
      vsSqrt(d, x, x);
    }

    template<>
    void sqrt(int d, double* x) {
      vdSqrt(d, x, x);
    }
    


    /**
     * Computes the operation \f$x[i]\gets x[i]^2\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x.
     * @param x Array to modify in-place.
     * @tparam T Data type.
     */
    template<typename T>
    void sqr(int d, T* x);

    template<>
    void sqr(int d, float* x) {
      vsSqr(d, x, x);
    }

    template<>
    void sqr(int d, double* x) {
      vdSqr(d, x, x);
    }



    /**
     * Computes the operation \f$y[i]\gets x[i]^2\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x and y.
     * @param x Input array.
     * @param y Output array.
     * @tparam T Data type.
     */
    template<typename T>
    void sqr(int d, const T* x, T* y);

    template<>
    void sqr(int d, const float* x, float* y) {
      vsSqr(d, x, y);
    }
    
    template<>
    void sqr(int d, const double* x, double* y) {
      vdSqr(d, x, y);
    }


    
    /**
     * Returns the Euclidean distance \f$||x-y||_2\f$.
     * @param d The length of x and y.
     * @param x Input array.
     * @param y Input array.
     * @param scratch An auxiliary array of length d.
     * @return \f$||x-y||_2\f$
     * @tparam T Data type.
     */
    template<typename T>
    T euclideanDistance(int d, const T* x, const T* y, T* scratch);

    template<>
    float euclideanDistance(int d, const float* x, const float* y,
			    float* scratch) {
      vsSub(d, x, y, scratch);
      return cblas_snrm2(d, scratch, 1);
    }

    template<>
    double euclideanDistance(int d, const double* x, const double* y,
			     double* scratch) {
      vdSub(d, x, y, scratch);
      return cblas_dnrm2(d, scratch, 1);
    }



    /**
     * Fills the array with 1s.
     * @param d The length of x.
     * @param x Array to modify in-place.
     * @tparam T Data type.
     */
    template<typename T>
    void ones(int d, T* x) {
      for (int i = 0; i < d; ++i)
	x[i] = 1;
    }
   



    /**
     * Returns the taxicab distance \f$||x-y||_1\f$.
     * @param d The length of x and y.
     * @param x Input array.
     * @param y Input array.
     * @param scratch An auxiliary array of length d.
     * @return \f$||x-y||_1\f$
     * @tparam T Data type.
     */
    template<typename T>
    T taxicabDistance(int d, const T* x, const T* y, T* scratch);

    template<>
    float taxicabDistance(int d, const float* x, const float* y, float* scratch) {
      vsSub(d, x, y, scratch);
      return cblas_sasum(d, scratch, 1);
    }

    template<>
    double taxicabDistance(int d, const double* x, const double* y, double* scratch) {
      vdSub(d, x, y, scratch);
      return cblas_dasum(d, scratch, 1);
    }



    /**
     * Computes the operation \f$D[i,j] \gets ||Q[i,:]-X[j,:]||_1\f$
     * for \f$i=0,1,\ldots,m-1\f$ and \f$j=0,1,\ldots,n-1\f$.
     * @param n The number of rows in X.
     * @param m The number of rows in Q.
     * @param d The number of columns in X and Q.
     * @param X Input array of shape \f$n\times d\f$.
     * @param Q Input array of shape \f$m\times d\f$.
     * @param dists Output array of shape \f$m\times n\f$
     * @param scratch An auxiliary array of length d.
     * @tparam T Data type.
     */
    template<typename T>
    void taxicabDistances(int n, int m, int d, const T* X, const T* Q,
			  T* dists, T* scratch) {
      for (int i = 0; i < m; ++i) {
	const T* q = Q + i*d;
	for (int j = 0; j < n; ++j) {
	  const T* x = X + j*d;
	  dists[i*n + j] = array::taxicabDistance(d, q, x, scratch);
	}
      }
    }

    
    /**
     * Computes the operation \f$x[i]\gets x[i] + y[i]\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x and y.
     * @param x Array to modify in place.
     * @param y Input array.
     * @tparam T Data type.
     */
    template<typename T>
    void add(int d, T* x, const T* y);
    
    
    
    template<>
    void add(int d, float* x, const float* y) {
      cblas_saxpy(d, 1.0f, y, 1, x, 1);
    }

    template<>
    void add(int d, double* x, const double* y) {
      cblas_daxpy(d, 1.0f, y, 1, x, 1);
    }



    /**
     * Computes the operation \f$x[i]\gets x[i] + ay[i]\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x and y.
     * @param x Array to modify in place.
     * @param a A scalar constant.
     * @param y Input array.
     * @tparam T Data type.
     */
    template<typename T>
    void add(int d, T* x, T a, const T* y);

    template<>
    void add(int d, float* x, float a, const float* y) {
      cblas_saxpy(d, a, y, 1, x, 1);
    }

    template<>
    void add(int d, double* x, double a, const double* y) {
      cblas_daxpy(d, a, y, 1, x, 1);
    }

    

    /**
     * Computes the operation \f$A\gets \alpha xy^\top + A\f$.
     * @param m The length of x.
     * @param n The length of y.
     * @param alpha A scalar constant.
     * @param x Input vector of shape \f$m\times 1\f$.
     * @param y Input vector of shape \f$n\times 1\f$.
     * @param A Output matrix modified in place of shape \f$n\times m\f$.
     * @tparam T Data type.
     */
    template<typename T>
    void ger(int m, int n, T alpha, const T* x, const T* y, T* A);

    template<>
    void ger(int m, int n, float alpha, const float* x,
	     const float* y, float* A) {
      cblas_sger(CblasRowMajor, m, n, alpha, x, 1, y, 1, A, n);
    }

    template<>
    void ger(int m, int n, double alpha, const double* x,
	     const double* y, double* A) {
      cblas_dger(CblasRowMajor, m, n, alpha, x, 1, y, 1, A, n);
    }

    
    
    /**
     * Computes the operation \f$z[i]\gets x[i] + y[i]\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x, y and z.
     * @param x Input array.
     * @param y Input array.
     * @param z Output array.
     * @tparam T Data type.
     */
    template<typename T>
    void add(int d, const T* __restrict x, const T* __restrict y, T* __restrict z) {
      for (int i = 0; i < d; ++i)
	z[i] = x[i] + y[i];
    }


    
    template<typename T>
    void add(int d, const T* __restrict x, T y, T* __restrict z) {
      for (int i = 0; i < d; ++i)
	z[i] = x[i] + y;
    }


    
    template<typename T>
    void add(int d, T* __restrict x, T y) {
      for (int i = 0; i < d; ++i)
	x[i] += y;
    }


    
    /**
     * Applies the unary function \f$f\f$ in place: \f$x[i] \gets
     * f(x[i])\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x.
     * @param x Array to modify in place.
     * @param f A function.
     * @tparam T Data type.
     * @tparam UnaryFunction Type of the function.
     */
    template<typename T, typename UnaryFunction>
    void unary(int d, T* x, UnaryFunction f) {
      for (int i = 0; i < d; ++i)
	x[i] = f(x[i]);
    }


    /**
     * Returns the sum of x: \f$\sum_{i=0}^{d-1} x[i]\f$.
     * @param d The length of x.
     * @param x Input array.
     * @return \f$\sum_{i=0}^{d-1} x[i]\f$
     * @tparam T Data type.
     */
    template<typename T>
    T sum(int d, const T* x) {
      T S = 0;
      for (int i = 0; i < d; ++i)
	S += x[i];
      return S;
    }


    
    /**
     * Returns the mean of x: \f$\frac{1}{d}\sum_{i=0}^{d-1} x[i]\f$.
     * @param d The length of x.
     * @param x Input array.
     * @return \f$\frac{1}{d}\sum_{i=0}^{d-1} x[i]\f$
     * @tparam T Data type.
     */
    template<typename T>
    T mean(int d, const T* x) {
      return sum(d,x)/d;
    }



    /**
     * Computes the operation \f$C\gets AB^\top\f$.
     * @param n The number of rows in A.
     * @param m The number of rows in B.
     * @param d The number of columns in A and B.
     * @param A Input matrix of shape \f$n\times d\f$.
     * @param B Input matrix of shape \f$m\times d\f$.
     * @param C Output matrix of shape \f$n\times m\f$.
     * @tparam T Data type.
     */
    template<typename T>
    void matmulT(int n, int m, int d, const T* __restrict A,
		 const T* __restrict B, T* __restrict C) {
      for (int i = 0; i < n; ++i) {
	for (int j = 0; j < m; ++j) {
	  const T* __restrict a = A + i*d;
	  const T* __restrict b = B + j*d;
	  T c = 0;
	  for (int k = 0; k < d; ++k) {
	    c += *a++ * *b++;
	  }
	  *C++ = c;
	}
      }
    }

    

    /**
     * Computes the General Matrix Multiply operation \f$C\gets \alpha
     * \mathrm{op}(A)\mathrm{op}(B) + \beta C\f$ where
     * \f$\mathrm{op}(X)=X^\top\f$ if the corresponding transposition
     * flag is set, and identity otherwise.
     * @param TA Whether to transpose A.
     * @param TB Whether to transpose B.
     * @param n The number of rows in A.
     * @param m The number of rows in B.
     * @param d The number of columns in A and B.
     * @param alpha Scalar constant.
     * @param A Input matrix of shape \f$n\times d\f$.
     * @param B Input matrix of shape \f$m\times d\f$.
     * @param beta Scalar constant.
     * @param C Output matrix of shape \f$n\times m\f$.
     * @tparam T Data type.
     */
    template<typename T>
    void gemm(bool TA, bool TB, int n, int m, int d, T alpha, const T* A,
	      const T* B, T beta, T* C);


    
    template<>
    inline void gemm(bool TA, bool TB, int n, int m, int d, float alpha,
		     const float* A, const float* B, float beta, float* C) {
      cblas_sgemm(CblasRowMajor, TA ? CblasTrans : CblasNoTrans,
		  TB ? CblasTrans : CblasNoTrans, n, m, d, alpha, A, 
		  TA ? n : d, B, TB ? d : m, beta, C, m);
    }



    template<>
    inline void gemm(bool TA, bool TB, int n, int m, int d, double alpha,
		     const double* A, const double* B, double beta, double* C) {
      cblas_dgemm(CblasRowMajor, TA ? CblasTrans : CblasNoTrans,
		  TB ? CblasTrans : CblasNoTrans, n, m, d, alpha, A, 
		  TA ? n : d, B, TB ? d : m, beta, C, m);
    }
    


    /**
     * Computes the General Matrix-Vector Multiply operation \f$y\gets \alpha
     * Ax + \beta y\f$.
     * @param n The number of rows in A.
     * @param d The number of columns in A.
     * @param alpha Scalar constant.
     * @param A Input matrix of shape \f$n\times d\f$.
     * @param x Input vector of shape \f$d\times 1\f$.
     * @param beta Scalar constant.
     * @param y Vector of shape \f$n\times 1\f$ modified in place.
     * @tparam T Data type.
     */
    template<typename T>
    void gemv(int n, int d, T alpha, const T* A,
	      const T* x, T beta, T* y);

    template<>
    inline void gemv(int n, int d, float alpha, const float* A, const float* x,
		     float beta, float* y) {
      cblas_sgemv(CblasRowMajor, CblasNoTrans, n, d, alpha, A, d, x, 1, beta,
		  y, 1);
    }

    template<>
    inline void gemv(int n, int d, double alpha, const double* A, const double* x,
		     double beta, double* y) {
      cblas_dgemv(CblasRowMajor, CblasNoTrans, n, d, alpha, A, d, x, 1, beta,
		  y, 1);
    }


    /**
     * Computes the operation \f$y[i]\gets x[i]\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x and y.
     * @param x Input array.
     * @param y Output array.
     * @tparam T Data type.
     */
    template<typename T>
    void mov(int d, const T* x, T* y);

    template<>
    void mov(int d, const float* x, float* y) {
      cblas_scopy(d, x, 1, y, 1);
    }



    template<>
    void mov(int d, const double* x, double* y) {
      cblas_dcopy(d, x, 1, y, 1);
    }



    /**
     * Computes the operation \f$x[i]\gets c\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x.
     * @param x Array to modify in place.
     * @param c Scalar constant.
     * @tparam T Data type.
     */
    template<typename T>
    void mov(int d, T* x, T c) {
      for (int i = 0; i < d; ++i)
	x[i] = c;
    }



    // required scratch: m + max(m,n)
    /**
     * Computes the squared Euclidean distances between two sets of
     * vectors: \f$D[i,j] \gets ||X[j,:] - Q[i,:]||_2^2\f$ for
     * \f$i=0,1,\ldots,n-1\f$ and \f$j=0,1,\ldots,m-1\f$.
     * @param n The number of rows in X.
     * @param m The number of rows in Q.
     * @param d The number of columns in X and Q.
     * @param X Input matrix of shape \f$n\times d\f$.
     * @param Q Input matrix of shape \f$m\times d\f$.
     * @param XSqNorms An auxiliary vector of length \f$n\f$ such that
     * the \f$i\f$th element is equal to \f$||X[i,:]||_2^2\f$.
     * @param dists Output matrix of shape \f$m\times n\f$.
     * @param scratch Auxiliary array of length \f$m + \max(m,n)\f$.
     * @tparam T Data type.
     */
    template<typename T>
    void euclideanSqDistances(int n, int m, int d, const T* X, const T* Q,
			      const T* XSqNorms, T* dists, T* scratch) {
      if (d > 0) {
	T* QSqNorms = scratch;
	T* ones = scratch + m;
	array::ones(std::max(m,n), ones);
	sqNorm(m, d, Q, QSqNorms);
	gemm(false, true, m, n, d, static_cast<T>(-2), Q, X,
	     static_cast<T>(0), dists);
	ger(m, n, static_cast<T>(1), QSqNorms, ones, dists);
	ger(m, n, static_cast<T>(1), ones, XSqNorms, dists);
	maxZero(m*n, dists);
      }
      else {
	mov(m*n, dists, static_cast<T>(0));
      }
    }



    /**
     * Computes the squared Euclidean distances from one vector to a set of vectors:
     \f$D[i] \gets ||X[i,:] - q||_2^2\f$ for
     * \f$i=0,1,\ldots,n-1\f$.
     * @param n The number of rows in X.
     * @param d The number of columns in X and the length of q.
     * @param X Input matrix of shape \f$n\times d\f$.
     * @param q Input vector of length \f$d\f$.
     * @param XSqNorms An auxiliary vector of length \f$n\f$ such that
     * the \f$i\f$th element is equal to \f$||X[i,:]||_2^2\f$.
     * @param dists Output array of length \f$n\f$.
     * @tparam T Data type.
     */
    template<typename T>
    void euclideanSqDistances(int n, int d, const T* __restrict X,
			      const T* __restrict q,
			      const T* __restrict XSqNorms,
			      T* __restrict dists) {
      if (d > 0) {
	T qSqNorm = sqNorm(d, q);
	add(n, XSqNorms, qSqNorm, dists);
	gemv(n, d, static_cast<T>(-2), X, q, static_cast<T>(1), dists);
	maxZero(n, dists);
      }
      else {
	mov(n, dists, static_cast<T>(0));
      }
    }



    /**
     * Returns true if and only if x and y compare equal, that is,
     * \f$x[i]=y[i]]\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x and y.
     * @param x Input array.
     * @param y Input array.
     * @return True if and only if x and y are equal.
     * @tparam T Data type.
     */
    template<typename T>
    bool equal(int d, const T* x, const T* y) {
      for (int i = 0; i < d; ++i)
	if (x[i] != y[i])
	  return false;
      return true;
    }
    

    
    /**
     * Returns true if and only if x and y compare almost equal, that is,
     * \f$|x[i]-y[i]| \leq \epsilon\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x and y.
     * @param x Input array.
     * @param y Input array.
     * @param y Scalar tolerance parameter.
     * @return True if and only if x and y are almost equal.
     * @tparam T Data type.
     */
    template<typename T>
    bool almostEqual(int d, const T* x, const T* y, T epsilon) {
      for (int i = 0; i < d; ++i)
	if (std::abs(x[i] - y[i]) > epsilon)
	  return false;
      return true;
    }



    /**
     * Returns true if and only if x and y compare almost equal, that is,
     * \f$\frac{|x[i]-y[i]|}{\max(x[i],y[i])} \leq \epsilon\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x and y.
     * @param x Input array.
     * @param y Input array.
     * @param y Scalar tolerance parameter.
     * @return True if and only if x and y are almost equal.
     * @tparam T Data type.
     */
    template<typename T>
    bool almostEqualRelative(int d, const T* x, const T* y, T epsilon) {
      for (int i = 0; i < d; ++i)
	if (std::abs((x[i] - y[i])/std::max(x[i],y[i])) > epsilon)
	  return false;
      return true;
    }



    /**
     * Prints the element in the given FILE stream in an appropriate precision.
     * @param elem An element.
     * @param f A FILE* pointer.
     * @tparam T Data type.
     */
    template<typename T>
    void printElem(T elem, FILE* f);

    template<>
    inline void printElem(float elem, FILE* f) {
      fprintf(f, "%.9f", elem);
    }
    
    template<>
    inline void printElem(double elem, FILE* f) {
      fprintf(f, "%.17f", elem);
    }
    

    /**
     * Prints the array in the given FILE stream in an appropriate precision.
     * @param n Number of rows in the array.
     * @param d Number of columns in the array.
     * @param X The array.
     * @param f A FILE* pointer.
     * @tparam T Data type.
     */
    template<typename T>
    void printArray(int n, int d, T* X, FILE* f) {
      fprintf(f, "[");
      for (int i = 0; i < n; ++i) {
	fprintf(f, "[");
	for (int j = 0; j < d; ++j) {
	  printElem(X[i*d+j], f);
	  fprintf(f, "%s", j < d-1 ? ", " : "");
	}
	fprintf(f, "]%s", i < n-1 ? ",\n" : "");
      }
      fprintf(f, "]\n");
    }

    

    /**
     * Prints the array in the given FILE stream in an appropriate
     * precision in CSV format.
     * @param n Number of rows in the array.
     * @param d Number of columns in the array.
     * @param X The array.
     * @param f A FILE* pointer.
     * @tparam T Data type.
     */
    template<typename T>
    void printCsv(int n, int d, T* X, FILE* f) {
      for (int i = 0; i < n; ++i) {
	for (int j = 0; j < d; ++j) {
	  printElem(X[i*d+j], f);
	  fprintf(f, "%s", j < d-1 ? "," : "");
	}
	fprintf(f, "\n");
      }
    }



    /**
     * Splits the data in half such that the value pointed to at m1 is
     * set to approximately half of n, m2 its complement, and the data
     * pointers Y and Z are set to point at appropriate locations
     * within the input array X.
     * @param n Number of rows in the array.
     * @param d Number of columns in the array.
     * @param m1 Pointer to the location where to store the number of
     * rows in the first half.
     * @param m2 Pointer to the location where to store the number of
     * rows in the second half.
     * @param X The input array.
     * @param Y Pointer to the location where to store the start of
     * the first half array.
     * @param Z Pointer to the location where to store the start of
     * the second half array.
     * @tparam T Data type.
     */
    template<typename T>
    void splitInHalf(int n, int d, int* m1, int* m2, T* X, T** Y, T** Z) {
      *m1 = n / 2;
      *m2 = n - *m1;
      *Y = X;
      *Z = X + *m1*d;
    }



    /**
     * Multiplies the array by constant: \f$x\gets cx\f$.
     * @param d The length of x and y.
     * @param x Array to modify in place.
     * @param c A scalar constant.
     * @tparam T Data type.
     */
    template<typename T>
    void mul(int d, T* x, T c);
    
    template<>
    void mul(int d, float* x, float c) {
      cblas_sscal(d, c, x, 1);
    }

    template<>
    void mul(int d, double* x, double c) {
      cblas_dscal(d, c, x, 1);
    }


    
    /**
     * Multiplies the array by constant: \f$y\gets cx\f$.
     * @param d The length of x and y.
     * @param x Input array.
     * @param y Output array.
     * @param c A scalar constant.
     * @tparam T Data type.
     */
    template<typename T>
    void mul(int d, const T* __restrict x, T* __restrict y, T c) {
      for (int i = 0; i < d; ++i)
	y[i] = x[i]*c;
    }


    
    /**
     * Computes the rowwise sum of \f$X\f$: \f$y[i]\gets \sum_{j=1}^d X[i,j]\f$.
     * @param n The number of rows in X.
     * @param d The number of columns in X.
     * @param X Input matrix of shape \f$n\times d\f$.
     * @param y Output vector of length \f$n\f$.
     * @tparam T Data type.
     */
    template<typename T>
    void rowwiseSum(int n, int d, const T* __restrict X, T* __restrict y) {
      for (int i = 0; i < n; ++i) {
	y[i] = 0;
	for (int j = 0; j < d; ++j) {
	  y[i] += X[i*d+j];
	}
      }
    }



    /**
     * Computes the rowwise mean of \f$X\f$: \f$y[i]\gets \frac{1}{d}\sum_{j=1}^d X[i,j]\f$.
     * @param n The number of rows in X.
     * @param d The number of columns in X.
     * @param X Input matrix of shape \f$n\times d\f$.
     * @param y Output vector of length \f$n\f$.
     * @tparam T Data type.
     */
    template<typename T>
    void rowwiseMean(int n, int d, const T* __restrict X, T* __restrict y) {
      for (int i = 0; i < n; ++i) {
	y[i] = 0;
	for (int j = 0; j < d; ++j) {
	  y[i] += X[i*d+j];
	}
	y[i] /= d;
      }
    }



    /**
     * Computes the operation \f$x[i]\gets \exp(x[i])\f$ for \f$i=0,1,\ldots,d-1\f$.
     * @param d The length of x.
     * @param x Array to modify in-place.
     * @tparam T Data type.
     */
    template<typename T>
    void exp(int d, T* x);



    template<>
    void exp(int d, float* x) {
      vsExp(d, x, x);
    }

    template<>
    void exp(int d, double* x) {
      vdExp(d, x, x);
    }



    /**
     * Computes the distance between two vectors under a metric specified at compile time.
     * @param d The length of x and y.
     * @param x Input array of length d.
     * @param y Input array of length d.
     * @param scratch Auxiliary array of length d.
     * @tparam T Data type.
     * @tparam Metric Metric to consider.
     */
    template<typename T, Metric M>
    T distance(int d, const T* x, const T* y, T* scratch);

    template<>
    float distance<float,Metric::EUCLIDEAN>(int d, const float* x,
					    const float* y, float* scratch) {
      return euclideanDistance(d,x,y,scratch);
    }

    template<>
    double distance<double,Metric::EUCLIDEAN>(int d, const double* x,
					    const double* y, double* scratch) {
      return euclideanDistance(d,x,y,scratch);
    }

    template<>
    float distance<float,Metric::TAXICAB>(int d, const float* x,
					    const float* y, float* scratch) {
      return taxicabDistance(d,x,y,scratch);
    }

    template<>
    double distance<double,Metric::TAXICAB>(int d, const double* x,
					    const double* y, double* scratch) {
      return taxicabDistance(d,x,y,scratch);
    }

    

    /**
     * Given an array of integer indices, computes the distance from a
     * query vector to the subset of vectors in a dataset, as
     * designated by the index: \f$Y[i]\gets D(X[idx[i],:],q)\f$ for
     * \f$i=0,1,\ldots,m-1\f$ where D is the distance function as
     * determined by the metric M.
     * @param m The number of elements in idx.
     * @param d The number of columns in X and the length of q.
     * @param X Input matrix with d columns and at least as many
     * rows as the largest element of idx.
     * @param idx Integral indices determining a subset of X.
     * @param Y Output array of length \f$m\f$.
     * @param scratch Auxiliary array of length d.
     * @tparam T Data type.
     * @taparm S Index data type.
     * @tparam M Metric to consider.
     */
    template<typename T, typename S, Metric M>
    void computeDistsSimple(int m, int d, const T* X, const T* q,
		      const S* idx, T* Y, T* scratch) {
      const T* x;
      for (int i = 0; i < m; ++i) {
	x = X + idx[i]*d;
	Y[i] = distance<T,M>(d,x,q,scratch);
      }
    }



    // required scratch size:
    // m*d + m + 1 + m
    /**
     * Given an array of integer indices, computes the Euclidean
     * distance from a query vector using matrix multiplication as
     * backend: \f$Y[i] \gets ||X[idx[i],:]-q||_2\f$ for
     * \f$i=0,1,\ldots,m-1\f$.
     * @param m The number of elements in idx.
     * @param d The number of columns in X and the length of q.
     * @param X Input matrix with d columns and at least as many
     * rows as the largest element of idx.
     * @param idx Integral indices determining a subset of X.
     * @param Y Output array of length \f$m\f$.
     * @param scratch Auxiliary array of length \f$m(d+2)+1\f$.
     * @tparam T Data type.
     * @taparm S Index data type.
     */
    template<typename T, typename S>
    void computeDistsEuclideanMatmul(int m, int d, const T* X, const T* q,
				     const S* idx, T* Y, T* scratch) {
      T* samples = scratch;
      scratch += m*d;
      T* sampleSqNorms = scratch;
      scratch += m;
      for (int i = 0; i < m; ++i) {
	const T* x = X + idx[i]*d;
	T* y = samples + i*d;
	mov(d, x, y);
      }
      sqNorm(m, d, samples, sampleSqNorms);
      euclideanSqDistances(m, 1, d, samples, q, sampleSqNorms, Y, scratch);
      sqrt(m, Y);
    }


    
    /**
     * Given an array of integer indices, computes the distance from a
     * query vector to the subset of vectors in a dataset, as
     * designated by the index: \f$Y[i]\gets D(X[idx[i],:],q)\f$ for
     * \f$i=0,1,\ldots,m-1\f$ where D is the distance function as
     * determined by the metric M.
     * @param m The number of elements in idx.
     * @param d The number of columns in X and the length of q.
     * @param X Input matrix with d columns and at least as many
     * rows as the largest element of idx.
     * @param idx Integral indices determining a subset of X.
     * @param Y Output array of length \f$m\f$.
     * @param scratch Auxiliary array of length d.
     * @tparam T Data type.
     * @taparm S Index data type.
     * @tparam M Metric to consider.
     */
    template<typename T, typename S, Metric M>
    void computeDists(int m, int d, const T* X, const T* q,
		      const S* idx, T* Y, T* scratch) {
      computeDistsSimple<T,S,M>(m, d, X, q, idx, Y, scratch);
    }



    /**
     * Returns true if and only if all elements of x compare equal to c.
     * @param d The length of x.
     * @param x Input array.
     * @param c Scalar constant.
     * @return True if and only if \f$x[i] = c\f$ for all \f$i=0,1,\ldots,d-1\f$.
     * @tparam T Data type.
     */
    template<typename T>
    bool allEqual(int d, const T* x, T c) {
      for (int i = 0; i < d; ++i)
	if (x[i] != c)
	  return false;
      return true;
    }



    /**
     * Returns a string that describes the VML modes for the MKL library.
     * @return A descriptory string.
     */
    inline std::string getVmlModeString() {
      unsigned int mod = vmlGetMode();
      unsigned int acc = mod & VML_ACCURACY_MASK;
      unsigned int den = mod & VML_FTZDAZ_MASK;
      unsigned int err = mod & VML_ERRMODE_MASK;
      std::vector<std::string> modes;
      if (acc & VML_LA)
        modes.push_back("VML_LA");
      if (acc & VML_HA)
        modes.push_back("VML_HA");
      if (acc & VML_EP)
        modes.push_back("VML_EP");
      if (den & VML_FTZDAZ_ON)
        modes.push_back("VML_FTZDAZ_ON");
      if (den & VML_FTZDAZ_OFF)
        modes.push_back("VML_FTZDAZ_OFF");
      if (err & VML_ERRMODE_NOERR)
        modes.push_back("VML_ERRMODE_NOERR");
      if (err & VML_ERRMODE_IGNORE)
        modes.push_back("VML_ERRMODE_IGNORE");
      if (err & VML_ERRMODE_ERRNO)
        modes.push_back("VML_ERRMODE_ERRNO");
      if (err & VML_ERRMODE_STDERR)
        modes.push_back("VML_ERRMODE_STDERR");
      if (err & VML_ERRMODE_EXCEPT)
        modes.push_back("VML_ERRMODE_EXCEPT");
      if (err & VML_ERRMODE_CALLBACK)
        modes.push_back("VML_ERRMODE_CALLBACK");
      if (err & (VML_ERRMODE_DEFAULT))
        modes.push_back("VML_ERRMODE_DEFAULT");
      if (modes.size() == 0)
        return "";
      std::string modeString = modes[0];
      for (auto it = modes.begin() + 1; it != modes.end(); ++it)
        modeString += " | " + *it;
      return modeString;
    }
  }
}

#endif // DEANN_ARRAY
