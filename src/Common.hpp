#ifndef DEANN_COMMON
#define DEANN_COMMON

#include "Array.hpp"
#include "Kernel.hpp"
#include <cassert>

/**
 * @file
 * This file contains some common subroutines.
 */


/**
 * This namespace contains all libkde content.
 */
namespace deann {
  using array::int_t;
  
  /**
   * Given \f$m\f$ distances, evaluates the KDE for the given kernel.
   * 
   * @param m Number of points
   * @param h Bandwidth \f$h>0\f$
   * @param dists Array of \f$m\f$ distances
   * @param scratch Scratch array of length \f$m\f$
   * @tparam T Datatype
   * @tparam K The kernel
   */
  template<typename T, Kernel K>
  T kdeDists(int_t m, T h, const T* dists, T* scratch) {
    if (K == Kernel::GAUSSIAN) {
      array::sqr(m, dists, scratch);
      array::mul(m, scratch, static_cast<T>(-0.5) / (h*h));
    }
    else {
      array::mul(m, dists, scratch, static_cast<T>(-1.0) / h);
    }
      
    array::exp(m, scratch);

    T Z = array::sum(m, scratch);
      
    return Z / m;
  }



  /**
   * Given \f$m\f$ distances, evaluates the KDE for the given kernel.
   * 
   * @param m Number of points
   * @param h Bandwidth \f$h>0\f$
   * @param dists Array of \f$m\f$ distances
   * @param scratch Scratch array of length \f$m\f$
   * @param K The kernel
   * @tparam T Datatype
   */
  template<typename T>
  T kdeDists(int_t m, T h, const T* dists, T* scratch, Kernel K) {
    switch(K) {
    case Kernel::EXPONENTIAL:
      return kdeDists<T,Kernel::EXPONENTIAL>(m,h,dists,scratch);
      break;
    case Kernel::GAUSSIAN:
      return kdeDists<T,Kernel::GAUSSIAN>(m,h,dists,scratch);
      break;
    case Kernel::LAPLACIAN:
      return kdeDists<T,Kernel::LAPLACIAN>(m,h,dists,scratch);
      break;
    }
    assert(false && "This line should never be reached");
    return 0; // to suppress warnings
  }



  /**
   * Given \f$m\f$ distances, evaluates the KDE for the given kernel
   * in place, modifying the input.
   * 
   * @param m Number of points
   * @param h Bandwidth \f$h>0\f$
   * @param scratch Scratch array of length \f$m\f$, prefilled with the distances.
   * @tparam T Datatype
   * @tparam K The kernel
   */
  template<typename T, Kernel K>
  T kdeDistsInPlace(int_t m, T h, T* scratch) {
    if (K == Kernel::GAUSSIAN) {
      array::sqr(m, scratch);
      array::mul(m, scratch, static_cast<T>(-0.5) / (h*h));
    }
    else {
      array::mul(m, scratch, static_cast<T>(-1.0) / h);
    }
      
    array::exp(m, scratch);

    T Z = array::sum(m, scratch);
      
    return Z / m;
  }


  
  // required scratch: d+m
  /**
   * Given \f$m\f$ indices in the data points, evaluates the KDE for
   * the subset of points as indicated by the indices. That is, given
   * \f$I\subseteq [n]\f$ such that \f$|I|=m\f$, returns 
   * \f$\frac{1}{m}\sum_{i\in I}K_h(q,x_i)\f$.
   * 
   * @param m Number of points
   * @param d Length of vectors (number of columns in X)
   * @param h Bandwidth \f$h>0\f$
   * @param X Dataset
   * @param q Query vector
   * @param idx Index vector of length \f$m\f$
   * @param scratch Scratch array of length \f$d+m\f$, prefilled with the distances.
   * @tparam T Datatype
   * @tparam S Index datatype (integer type)
   * @tparam K The kernel
   */
  template<typename T, typename S, Kernel K>
  T kdeSubset(int_t m, int_t d, T h, const T* X, const T* q, S* idx,
	      T* scratch) {
    if (K == Kernel::LAPLACIAN)
      array::computeDists<T,S,Metric::TAXICAB>(m, d, X, q, idx, scratch,
					       scratch + m);
    else
      array::computeDists<T,S,Metric::EUCLIDEAN>(m, d, X, q, idx, scratch,
						 scratch + m);
    return kdeDistsInPlace<T, K>(m, h, scratch);
  }




  // required scratch: d+m
  /**
   * Given \f$m\f$ indices in the data points, evaluates the KDE for
   * the subset of points as indicated by the indices. That is, given
   * \f$I\subseteq [n]\f$ such that \f$|I|=m\f$, returns 
   * \f$\frac{1}{m}\sum_{i\in I}K_h(q,x_i)\f$.
   * 
   * @param m Number of points
   * @param d Length of vectors (number of columns in X)
   * @param h Bandwidth \f$h>0\f$
   * @param X Dataset
   * @param q Query vector
   * @param idx Index vector of length \f$m\f$
   * @param scratch Scratch array of length \f$d+m\f$, prefilled with the distances.
   * @param K The kernel
   * @tparam T Datatype
   * @tparam S Index datatype (integer type)
   */
  template<typename T, typename S>
  T kdeSubset(int_t m, int_t d, T h, const T* X, const T* q, S* idx,
	      T* scratch, Kernel K) {
    switch(K) {
    case Kernel::EXPONENTIAL:
      return kdeSubset<T,S,Kernel::EXPONENTIAL>(m,d,h,X,q,idx,scratch);
      break;
    case Kernel::GAUSSIAN:
      return kdeSubset<T,S,Kernel::GAUSSIAN>(m,d,h,X,q,idx,scratch);
      break;
    case Kernel::LAPLACIAN:
      return kdeSubset<T,S,Kernel::LAPLACIAN>(m,d,h,X,q,idx,scratch);
      break;
    }
    assert(false && "This line should never be reached");
    return 0; // to suppress warnings
  }
}

#endif // DEANN_COMMON

