#ifndef DEANN_KDE_ESTIMATOR
#define DEANN_KDE_ESTIMATOR

#include "FastRNG.hpp"
#include "Kernel.hpp"
#include <random>
#include <optional>


/**
 * @file
 * This file provides the abstract base class for estimators.
 */

namespace deann {
  using array::int_t;
  
  /**
   * Abstract base class for estimators. Subclasses provide actual functionality.
   */
  class KdeEstimator {
  public:
    /**
     * Pseudo random number generator typedef. Does not fully
     * correspond to standard library pseudo random number generators.
     */
    typedef FastRng PseudoRandomNumberGenerator;



    /**
     * Compulsory pure virtual destructor.
     */
    virtual ~KdeEstimator() = 0;



    /**
     * Reset the parameters of the instance. Trying to reset more
     * parameters than there are will result in an exception being
     * thrown.
     * 
     * @param param1 The first parameter whose meaning is determined
     * by the subclass.
     * @param param2 The second parameter whose meaning is determined
     * by the subclass.
     */
    virtual void resetParameters(std::optional<int_t> param1 = std::nullopt,
				 std::optional<int_t> param2 = std::nullopt) = 0;

    /**
     * Reset the random number seed. Functionality is imlemented in
     * the middle class. The function is provided here for interface
     * reasons.
     *
     * @param seed New seed (or nullopt for unpredicatble default 
     * initialization)
     */
    virtual void resetSeed(std::optional<PseudoRandomNumberGenerator::ValueType>
                           seed = std::nullopt) = 0;
    
  private:
  };

  

  inline KdeEstimator::~KdeEstimator() {
  }


  
  /**
   * This is a templatized middle class for providing interface for
   * both single and double precision floating point arithmetic.
   * @tparam T Type for the arithmetic to use, either float or double.
   */
  template<typename T>
  class KdeEstimatorT : public KdeEstimator {
  public:
    static_assert(std::is_floating_point<T>::value);



    /**
     * Middle-class constructor for managing compulsory parameters
     * common to all estimators.
     * @param bandwidth Bandwidth \f$h>0\f$
     * @param kernel Kernel to use (some subclasses may have
     * restrictions on which kernels they support)
     * @param Optional random number generator seed, or nullopt for 
     * unpredictable default initialization.
     */
    KdeEstimatorT(T bandwidth, Kernel kernel,
		  std::optional<PseudoRandomNumberGenerator::ValueType> seed =
		  std::nullopt) :
      rngSeed(seed ? *seed : std::random_device()()),
      rng(~0u, rngSeed) {
      setBandwidth(bandwidth);
      setKernel(kernel);
    }



    /**
     * Fit dataset, that is, copy pointer to X into the data
     * structure. The caller must ensure that the pointer X remains
     * valid for the entire lifetime of the estimator. Subclasses may
     * also perform some additional computation.
     * @param n Number of rows (data vectors) in the dataset.
     * @param d Number of columns in the dataset (dimension of individual vectors).
     * @param X Pointer to the dataset, interpeted as a n*d matrix, or
     * an array of length n*d.
     */
    void fit(int_t n, int_t d, const T* X) {
      if (n < 1 || d < 1)
	throw std::invalid_argument("Dataset cannot be empty (got n=" +
				    std::to_string(n) + ", d=" +
				    std::to_string(d) + ")");
      if (!X)
	throw std::invalid_argument("Dataset must be non-null");
      this->n = n;
      this->d = d;
      this->X = X;
      rng = PseudoRandomNumberGenerator(n,rngSeed);

      fitImpl();
    }



    /**
     * Query an individual vector. Actual computation depends on the subclass.
     * A dataset must be fitted before this function is called, otherwise an 
     * exception will be thrown.
     * @param d Length of the vector. Must match that of the fitted dataset.
     * @param q Pointer to the vector (an array of length d).
     * @param Z Pointer to a variable where the estimate will be stored.
     * @param samples Pointer to a variable where the  number of samples looked 
     * at will be stored (or null if the information is not required)
     */
    void query(int_t d, const T* q, T* Z, int_t* samples = nullptr) const {
      if (!X)
	throw std::invalid_argument("The object is uninitialized, please call "
				    "fit first.");
      if (d != this->d)
	throw std::invalid_argument("inconsistent dimensions (got " +
				    std::to_string(d) + " but expected " +
				    std::to_string(this->d) + ")");
      queryImpl(q, Z, samples);
    }



    /**
     * Perform a batch query on a number of vectors. Actual
     * computation depends on the subclass; the default is to perform
     * individual queries on each of the supplied vectors, but
     * subclasses may override this behavior. A dataset must be fitted
     * before this function is called, otherwise an exception will be
     * thrown.
     * @param m Number of query vectors.
     * @param d Length of the vectors. Must match that of the fitted dataset.
     * @param Q Pointer to the query matrix (an array of length m*d) whose rows 
     * are the query vectors.
     * @param Z Pointer to an array of length m where the KDE estimates are 
     * stored.
     * @param samples Pointer to an array of length m where the number
     * of samples looked at during the query are stored (or null if the 
     * information is not required).
     */
    void query(int_t m, int_t d, const T* Q, T* Z, int_t* samples = nullptr) const {
      if (!X)
	throw std::invalid_argument("The object is uninitialized, please call "
				    "fit first.");
      if (d != this->d)
	throw std::invalid_argument("inconsistent dimensions (got " +
				    std::to_string(d) + " but expected " +
				    std::to_string(this->d) + ")");
      if (m < 1)
	throw std::invalid_argument("Query matrix must be non-empty, got " +
				    std::to_string(m) + " rows");
      queryImpl(m, Q, Z, samples);
    }



    /**
     * Reseed the pseudorandom number generator with the desired seed,
     * or perform unpredictable default initialization if nullopt is
     * provided.
     * @param seed Optional seed value.
     */
    inline void resetSeed(std::optional<PseudoRandomNumberGenerator::ValueType> seed =
			  std::nullopt) override {
      rngSeed = seed ? *seed : std::random_device()();
      rng = PseudoRandomNumberGenerator(n > 0 ? n : 0, rngSeed);
    }


    
#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    /**
     * For debug purposes
     */
    KdeEstimator::PseudoRandomNumberGenerator& getRng() {
      return this->rng;
    }
#endif // DEANN_ENABLE_DEBUG_ACCESSORS

    
    
  protected:
    T h = 0;
    Kernel K = Kernel::EXPONENTIAL;
    int_t n = 0;
    int_t d = 0;
    const T* X = nullptr;
    PseudoRandomNumberGenerator::ValueType rngSeed;
    mutable PseudoRandomNumberGenerator rng;

  private:
    /**
     * Reset the kernel for the estimator.
     * @param kernel New kernel
     */
    void setKernel(Kernel kernel) {
      switch(kernel) {
      case Kernel::EXPONENTIAL:
      case Kernel::GAUSSIAN:
      case Kernel::LAPLACIAN:
	K = kernel;
	break;
      default:
	throw std::invalid_argument("Unsupported kernel supplied");
      }
    }



    /**
     * Reset the bandwidth.
     * @param bandwidth New bandwidth \f$h>0\f$.
     */
    void setBandwidth(T bandwidth) {
      if (bandwidth <= 0)
	throw std::invalid_argument("Bandwidth must be positive (got: " +
				    std::to_string(bandwidth) + ")");
      h = bandwidth;
    }


    
    virtual void queryImpl(const T* q, T* Z, int_t* samples) const = 0;
    virtual void queryImpl(int_t m, const T* Q, T* Z, int_t* samples) const = 0;
    virtual void fitImpl() = 0;
  };
}

#endif // DEANN_KDE_ESTIMATOR
