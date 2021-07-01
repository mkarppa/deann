#ifndef DEANN_NAIVE_KDE
#define DEANN_NAIVE_KDE

#include "KdeEstimator.hpp"
#include "Array.hpp"
#include "Kernel.hpp"
#include <memory>
#include <string>
#include <chrono>
#include <iostream>

/**
 * @file
 * This file contains the implementation of the naive KDE.
 */

namespace deann {
  /**
   * This function presents a simple naive KDE implementation that can
   * be used to compute the KDE values as a baseline, although at a
   * relatively poor performance.
   *
   * @param n Dataset vector count
   * @param m Query vector count
   * @param d Dimensionality
   * @param h Bandwidth
   * @param X Dataset as an \f$n\times d\f$ array
   * @param Q Queries as an \f$m\times d\f$ array
   * @param mu Output array of length \f$m\f$ for the KDE values.
   * @param scratch A scratch array of length \f$d\f$.
   * @param kernel Kernel
   * @tparam T Data type
   */
  template<typename T>
  void kdeEuclideanSimple(int n, int m, int d, T h, const T* __restrict X,
			  const T* __restrict Q, T* __restrict mu,
			  T* __restrict scratch,
			  Kernel kernel) {
    switch(kernel) {
    case Kernel::EXPONENTIAL:
    case Kernel::GAUSSIAN:
      break;
    default:
      throw std::invalid_argument("Unsupported kernel supplied");
    }
    for (int i = 0; i < m; ++i) {
      mu[i] = 0;
      const T* q = Q + i*d;
      for (int j = 0; j < n; ++j) {
	const T* x = X + j*d;
	T dist = array::euclideanDistance(d, q, x, scratch);
	T kv = 0;
	if (kernel == Kernel::EXPONENTIAL)
	  kv = std::exp(-dist/h);
	else if (kernel == Kernel::GAUSSIAN)
	  kv = std::exp(-dist*dist/h/h/2);
	mu[i] += kv;
      }
      mu[i] /= n;
    }
  }
 


  // required scratch size: n*m + m + max(n,m)
  /**
   * This function computes the naive KDE using matrix multiplication.
   *
   * @param n Dataset vector count
   * @param m Query vector count
   * @param d Dimensionality
   * @param h Bandwidth
   * @param X Dataset as an \f$n\times d\f$ array
   * @param Q Queries as an \f$m\times d\f$ array
   * @param mu Output array of length \f$m\f$ for the KDE values.
   * @param XSqNorms An array of length \f$n\f$, the ith element of
   * which should contain the squared Euclidean norm of the ith vector
   * in X.
   * @param scratch A scratch array of length \f$nm + m + max(n,m)\f$.
   * @param kernel Kernel (must be Euclidean)
   * @tparam T Data type
   */
  template<typename T>
  void kdeEuclideanMatmul(int n, int m, int d, T h, const T* __restrict X,
			  const T* __restrict Q, T* __restrict mu,
			  const T* __restrict XSqNorms, T* __restrict scratch,
			  Kernel kernel) {
    switch(kernel) {
    case Kernel::EXPONENTIAL:
    case Kernel::GAUSSIAN:
      break;
    default:
      throw std::invalid_argument("Unsupported kernel supplied");
    }
    
    if (m < 1)
      return;
  
    array::euclideanSqDistances(n, m, d, X, Q, XSqNorms, scratch, scratch + n*m);
    if (kernel == Kernel::EXPONENTIAL) {
      array::sqrt(m*n, scratch);
      array::mul(m*n, scratch, -static_cast<T>(1)/h);
    }
    else if (kernel == Kernel::GAUSSIAN) {
      array::mul(m*n, scratch, -static_cast<T>(1)/h/h/2);
    }
    array::exp(m*n, scratch);
    array::rowwiseMean(m, n, scratch, mu);
  }



  // required scratch size: n
  /**
   * This function computes the naive KDE using matrix multiplication
   * for a single query vector.
   *
   * @param n Dataset vector count
   * @param d Dimensionality
   * @param h Bandwidth
   * @param X Dataset as an \f$n\times d\f$ array
   * @param q Query as a \f$d\f$ array
   * @param XSqNorms An array of length \f$n\f$, the ith element of
   * which should contain the squared Euclidean norm of the ith vector
   * in X.
   * @param scratch A scratch array of length \f$n\f$.
   * @param kernel Kernel (must be Euclidean)
   * @tparam T Data type
   * @return The KDE value
   */
  template<typename T>
  T kdeEuclideanMatmul(int n, int d, T h, const T* __restrict X,
		       const T* __restrict q,  const T* __restrict XSqNorms,
		       T* __restrict scratch, Kernel kernel) {
    switch(kernel) {
    case Kernel::EXPONENTIAL:
    case Kernel::GAUSSIAN:
      break;
    default:
      throw std::invalid_argument("Unsupported kernel supplied");
    }
    
    array::euclideanSqDistances(n, d, X, q, XSqNorms, scratch);
    if (kernel == Kernel::EXPONENTIAL) {
      array::sqrt(n, scratch);
      array::mul(n, scratch, -static_cast<T>(1)/h);
    }
    else if (kernel == Kernel::GAUSSIAN) {
      array::mul(n, scratch, -static_cast<T>(1)/h/h/2);
    }
    array::exp(n, scratch);
    return array::mean(n, scratch);
  }




  /**
   * This function computes the naive KDE for the taxicab metrix (without matmul).
   *
   * @param n Dataset vector count
   * @param m Query vector count
   * @param d Dimensionality
   * @param h Bandwidth
   * @param X Dataset as an \f$n\times d\f$ array
   * @param Q Queries as an \f$m\times d\f$ array
   * @param mu Output array of length \f$m\f$ for the KDE values.
   * @param scratch A scratch array of length \f$mn+d\f$.
   * @param kernel Kernel (must be compatible with taxicab metric, so basically laplacian)
   * @tparam T Data type
   */
  template<typename T>
  void kdeTaxicab(int n, int m, int d, T h, const T* __restrict X,
		  const T* __restrict Q, T* __restrict mu,
		  T* __restrict scratch,
		  Kernel kernel) {
    switch(kernel) {
    case Kernel::LAPLACIAN:
      break;
    default:
      throw std::invalid_argument("Unsupported kernel supplied");
    }
    
    if (m < 1)
      return;

    T* __restrict taxiScratch = scratch + m*n;
    for (int i = 0; i < m; ++i) {
      const T* __restrict q = Q + i*d;
      for (int j = 0; j < n; ++j) {
	const T* __restrict x = X + j*d;
	scratch[i*n + j] = array::taxicabDistance(d, q, x, taxiScratch);
      }
    }
    array::mul(m*n, scratch, -static_cast<T>(1)/h);
    array::exp(m*n, scratch);
    array::rowwiseMean(m, n, scratch, mu);
  }



  /**
   * A class for computing the exact KDE value \f$\mathrm{KDE}_X(q) = \frac{1}{n}
   * \sum_{i=1}^n K_h(x_i,q) \f$ naively.
   */
  template<typename T>
  class NaiveKde : public KdeEstimatorT<T> {
  public: 
    /**
     * The usual constructor.
     * @param bandwidth Bandwidth
     * @param kernel The kernel function object \f$K_h\f$
     */
    NaiveKde(double bandwidth,
	     Kernel kernel) :
      KdeEstimatorT<T>(bandwidth, kernel) {
    }



    /**
     * Not used by the class. Throws an exception.
     */
    void resetParameters(std::optional<int> param1 = std::nullopt,
			 std::optional<int> param2 = std::nullopt) override {
      if (param1 || param2)
	throw std::invalid_argument("The NaiveKde class has no parameters to "
				    "set, but tried to reset parameters "
				    "anyway.");
    }


        
  private:
    void fitImpl() override {
      if (this->K == Kernel::EXPONENTIAL || this->K == Kernel::GAUSSIAN) {
	XSqNorm = std::vector<T>(this->n);
	array::sqNorm(this->n, this->d, this->X, &XSqNorm[0]);
      }
    }


    
    void queryImpl(const T* q, T* Z, int* samples) const override {
      if (this->X < q + this->d && q < this->X+this->n*this->d)
	throw std::invalid_argument("q breaks aliasing rules");
      
      T* scratch = nullptr;
      switch(this->K) {	
      case Kernel::EXPONENTIAL:
      case Kernel::GAUSSIAN:
	scratch = new T[this->n+1+std::max(this->n,1)];
	kdeEuclideanMatmul(this->n, 1, this->d, this->h, this->X, q, Z,
			   &XSqNorm[0], scratch, this->K);
	break;
	
      case Kernel::LAPLACIAN:
	scratch = new T[this->n+this->d];
	kdeTaxicab(this->n, 1, this->d, this->h, this->X, q, Z, scratch,
		   this->K);
	break;
      }
      delete[] scratch;

      if (samples)
        *samples = this->n;
    }


    
    void queryImpl(int m, const T* Q, T* Z, int* samples) const override {
      if ((this->X - this->d < Q && Q < this->X+this->n*this->d) ||
	  (this->X <= Q + m*this->d - 1 && Q + m*this->d - 1 < this->X +
	   this->n*this->d))
	throw std::invalid_argument("Q breaks aliasing rules");

      if (this->K == Kernel::EXPONENTIAL || this->K == Kernel::GAUSSIAN) {
	std::vector<T> scratch(m*this->n+m+std::max(this->n,m));      
	kdeEuclideanMatmul(this->n, m, this->d, this->h, this->X, Q, Z,
			   &XSqNorm[0], &scratch[0], this->K);
      }
      else if (this->K == Kernel::LAPLACIAN) {
	std::vector<T> scratch(m*this->n+this->d);
	kdeTaxicab(this->n, m, this->d, this->h, this->X, Q, Z, &scratch[0],
		   this->K);
      }
      if (samples)
        array::mov(m, samples, this->n);
    }


    
    std::vector<T> XSqNorm;
  };
}

#endif // DEANN_NAIVE_KDE
