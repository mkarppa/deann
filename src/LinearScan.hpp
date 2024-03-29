#ifndef DEANN_LINEAR_SCAN
#define DEANN_LINEAR_SCAN

#include "Array.hpp"
#include <vector>
#include <stdexcept>

/**
 * @file 
 * This file provides the implementation of the simple linear scan NN.
 */

namespace deann {
  using array::int_t;
  
  /**
   * This class implements a simple linear scan nearest neighbors algorithm.
   */
  template<typename T>
  class LinearScan {
  public:  
    /**
     * The simple constructor.
     *
     * @param metric The metric to use
     * @param reportDistances Whether to report distances upon query or just 
     * fill the vector with -1s.
     * @param reportSamples Whether to report the number of samples looked at 
     * or fill the vector with -1s.
     */
    explicit LinearScan(Metric metric, bool reportDistances = true,
                        bool reportSamples = true) :
      metric(metric), reportDistances(reportDistances),
      reportSamples(reportSamples) {
      switch (metric) {
      case Metric::EUCLIDEAN:
      case Metric::TAXICAB:
	break;
      }
    }



    /**
     * Fit the data, that is, copy the data into the data structure
     * for further use.
     *
     * @param dataset An \f$d\times n\f$ dataset of points. The number of 
     * columns \f$n\f$ must be greater than \f$k\f$ and the number of rows
     * \f$d\f$ must be positive.
     */
    void fit(int_t n, int_t d, const T* dataset) {
      if (n < 1)
        throw std::invalid_argument("Too few vectors: at least n=1 vector "
                                    "expected (got " + std::to_string(n) + ")");
      if (d < 1)
	throw std::invalid_argument("Too small dimension: at least d=1 "
				    "expected (got " + std::to_string(d) + ")");

      X = dataset;
      this->n = n;
      this->d = d;
      if (metric == Metric::EUCLIDEAN) {
	XSqNorms = std::vector<T>(n);
	array::sqNorm(n, d, X, &XSqNorms[0]);
      }
    }



    /**
     * Query the data structure for the k-nearest neighbors of the given vector.
     * If the number of neighbors requested \f$k\f$ is greater than the number
     * of vectors \f$n\f$ in the dataset, the remaining entries are filled with 
     * -1s.
     * 
     * @param m Number of query vectors
     * @param d Distance of query vectors, must correspond to the underlying 
     * dimension of the data set.
     * @param K The number of nearest neighbors to query, must satisfy \f$K>0\f$
     * @param Q The query vectors as an \f$m\times d\f$ matrix (a row-major array)
     * @param N Pointer to an array of length \f$m\times K\f$ where the indices 
     * of the nearest neighbors are stored in ascending order of distance
     * @param D Pointer to an array of length \f$m\times K\f$ where the 
     * distances of the nearest neighbors are stored in ascending order, or -1s
     * if the distances are not to be computed.
     * @param S Pointer to an array of length \f$m\f$ where the number of 
     * samples looked at per query vector is stored (will be filled with the 
     * number of points \f$n\f$ in the dataset, or -1s if this information is 
     * not requested.
     */
    void query(int_t m, int_t d, int_t K, const T* Q, int_t* N, T* D, int_t* S = nullptr) const {
      if (n < 1 || this->d < 1)
	throw std::invalid_argument("Data structure is uninitialized");
      if (d != this->d)
	throw std::invalid_argument("Dimension mismatch");
      if (m < 1)
	throw std::invalid_argument("Must query at least 1 vector");
      if (K < 1) {
        array::mov(m, S,
		   reportSamples ? static_cast<int_t>(0) : static_cast<int_t>(-1));
	return;
      }
      int_t k = std::min(n,K);

      try {
	std::vector<T> dists(m*n);
	std::vector<T> scratch(std::max(m+std::max(m,n), d));
	switch(metric) {
	case Metric::EUCLIDEAN:
	  array::euclideanSqDistances(n, m, d, X, Q, &XSqNorms[0], &dists[0],
				      &scratch[0]);
	  break;
	case Metric::TAXICAB:
	  array::taxicabDistances(n, m, d, X, Q, &dists[0], &scratch[0]);
	  break;
	}
	for (int_t l = 0; l < m; ++l) {
	  std::vector<std::pair<T,int_t>> distNns(k);
	  for (int_t i = 0; i < k; ++i)
	    distNns[i] = std::make_pair(dists[l*n+i], i);
	  std::sort(distNns.begin(), distNns.end());

	  for (int_t i = k; i < n; ++i) {
	    T dist = dists[l*n+i];
	    if (dist >= distNns[k-1].first)
	      continue;
	    int_t j = k-1;
	    while (j > 0 && dist < distNns[j-1].first) {
	      distNns[j] = distNns[j-1];
	      --j;
	    }
	    distNns[j] = std::make_pair(dist,i);
	  }
	  for (int_t i = 0; i < k; ++i) {
	    D[l*K + i] = reportDistances ? distNns[i].first : -1;
	    N[l*K + i] = distNns[i].second;
	  }
	  if (reportDistances && metric == Metric::EUCLIDEAN)
	    array::sqrt(k, D + l*K);
	  for (int_t i = k; i < K; ++i) {
	    D[l*K + i] = -1;
	    N[l*K + i] = -1;
	  }
	}
	if (S)
	  array::mov(m, S, reportSamples ? n : -1);
      }
      catch (std::bad_alloc& e) {
	throw std::runtime_error("Could not allocate enough scratch space; query "
				 "with n=" + std::to_string(n) +
				 ", m=" + std::to_string(m) +
				 ", d=" + std::to_string(d) +
				 " requires at least " +
				 std::to_string((m*n+std::max(m+std::max(m,n),d))*sizeof(T)) +
				 " bytes of allocatable memory.");
      }
    }

    

    void query(int_t d, int_t K, const T* Q, int_t* N, T* D, int_t* S = nullptr) const {
      query(1, d, K, Q, N, D, S);
    }


    
  private:    
    Metric metric = Metric::EUCLIDEAN;
    bool reportDistances = true;
    bool reportSamples = true;
    const T* X = nullptr;
    int_t n = 0;
    int_t d = 0;
    std::vector<T> XSqNorms;
  };
}

#endif // DEANN_LINEAR_SCAN
