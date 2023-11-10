#ifndef DEANN_RANDOM_SAMPLING
#define DEANN_RANDOM_SAMPLING

#include "Common.hpp"
#include "KdeEstimator.hpp"
#include "Array.hpp"
#include "Kernel.hpp"
#include <random>
#include <string>

/**
 * @file
 * This file provides the implementation of the simple RS estimator.
 */

namespace deann {
  /**
   * Computes the unbiased estimate \f$Z = \frac{1}{m}\sum_{x'\in X'} K_h(q,x')\f$
   * where the sequence \f$X'=(x'_0,x'_1,\ldots,x'_{m-1})\f$ is chosen uniformly 
   * and independently at random.   
   */
  template<typename T>
  class RandomSampling : public KdeEstimatorT<T> {
  public:
    /**
     * Basic constructor.
     * @param bandwidth Bandwidth
     * @param kernel The kernel
     * @param randomSamples The number of random samples \f$m>0\f$
     * @param seed Pseudorandom generator seed, or nullopt for unpredictable initialization
     */ 
    RandomSampling(T bandwidth, Kernel kernel, int_t randomSamples,
		   std::optional<KdeEstimator::PseudoRandomNumberGenerator::
		   ValueType> seed = std::nullopt) :
      KdeEstimatorT<T>(bandwidth, kernel, seed) {
      setRandomSamples(randomSamples);
    }

    

    /**
     * Resets the number of random samples
     * @param newSamples New number of random samples.
     */
    void setRandomSamples(int_t newSamples) {
      if (newSamples < 1)
	throw std::invalid_argument("The number of samples must be positive "
				    "(got: " + std::to_string(newSamples) + ")");
      randomSamples = newSamples;
      scratch = std::vector<T>(randomSamples + this->d);
      sampleIdx = std::vector<FastRng::ValueType>(randomSamples);
    }


    
    /**
     * Resets the number of random samples
     * @param param1 New number of random samples.
     * @param param2 Unused. Attempting to use throws an exception.
     */
    void resetParameters(std::optional<int_t> param1 = std::nullopt,
			 std::optional<int_t> param2 = std::nullopt) override {
      if (param2)
	throw std::invalid_argument("The RandomSampling class has only one "
				    "parameter, but tried to reset two "
				    "parameters.");
      if (param1)
	setRandomSamples(*param1);
    }



  private:
    void fitImpl() override {
      scratch = std::vector<T>(this->d + randomSamples);
      sampleIdx = std::vector<FastRng::ValueType>(randomSamples);
    }



    void queryImpl(const T* q, T* Z, int_t* samples) const override {
      this->rng(randomSamples, &sampleIdx[0]);

      if (this->K == Kernel::EXPONENTIAL) 
	*Z = kdeSubset<T,FastRng::ValueType,Kernel::EXPONENTIAL>(randomSamples, this->d, this->h,
                                                       this->X, q, &sampleIdx[0],
                                                       &scratch[0]);
      else if (this->K == Kernel::GAUSSIAN)
	*Z = kdeSubset<T,FastRng::ValueType,Kernel::GAUSSIAN>(randomSamples, this->d, this->h,
                                                    this->X, q, &sampleIdx[0],
                                                    &scratch[0]);
      else if (this->K == Kernel::LAPLACIAN)
	*Z = kdeSubset<T,FastRng::ValueType,Kernel::LAPLACIAN>(randomSamples, this->d, this->h,
                                                     this->X, q, &sampleIdx[0],
                                                     &scratch[0]);
      else
	throw std::invalid_argument("Implementation error, this shouldn't happen.");
      
      if (samples)
        *samples = randomSamples;
    }

    

    void queryImpl(int_t m, const T* Q, T* Z, int_t* samples) const override {
      for (int_t i = 0; i < m; ++i) 
        queryImpl(Q + i*this->d, Z + i, samples ? samples + i : nullptr);
    }



    int_t randomSamples = 0;
    mutable std::vector<T> scratch;
    mutable std::vector<FastRng::ValueType> sampleIdx;
  };
}

#endif // DEANN_RANDOM_SAMPLING
