#ifndef DEANN_RANDOM_SAMPLING_PERMUTED
#define DEANN_RANDOM_SAMPLING_PERMUTED

#include "NaiveKde.hpp"
#include "Array.hpp"
#include "KdeEstimator.hpp"
/**
 * @file
 * This file provides the implementation of the permuted RS estimator.
 */

namespace deann {
  /**
   * Computes the unbiased estimate \f$Z = \frac{1}{m}\sum_{x'\in X'} K_h(q,x')\f$
   * where the set \f$X'=\{x'_0,x'_1,\ldots,x'_{m-1}\}\f$ is determined during 
   * preprocessing by permuting the dataset at random. During queries, a contiguous 
   * subset of points is used, and if Euclidean metric is used, the estimate is 
   * computed using matrix multiplication.
   */
  template<typename T>
  class RandomSamplingPermuted : public KdeEstimatorT<T> {
  public:
    /**
     * Basic constructor.
     * @param bandwidth Bandwidth
     * @param kernel The kernel
     * @param randomSamples The number of random samples \f$m>0\f$
     * @param seed Pseudorandom generator seed, or nullopt for unpredictable initialization
     */ 
    RandomSamplingPermuted(T bandwidth, Kernel kernel, int randomSamples,
			   std::optional<KdeEstimator::
			   PseudoRandomNumberGenerator::ValueType> seed =
			   std::nullopt) :
      KdeEstimatorT<T>(bandwidth, kernel, seed) {
      if (kernel != Kernel::EXPONENTIAL && kernel != Kernel::GAUSSIAN)
	throw std::invalid_argument("Only exponential and gaussian kernels are "
				    "supported by this class.");
      setRandomSamples(randomSamples);
    }



    /**
     * Resets the number of random samples
     * @param newSamples New number of random samples.
     */
    void setRandomSamples(int newSamples) {
      if (newSamples < 1)
	throw std::invalid_argument("The number of samples must be positive "
				    "(got: " + std::to_string(newSamples) + ")");
      if (this->X && newSamples > this->n)
        throw std::invalid_argument("Cannot set random samples to " +
                                    std::to_string(newSamples) + " because it "
                                    "exceeds the size of the fitted dataset ("
                                    + std::to_string(this->n) + ")");
      randomSamples = newSamples;
      scratch = std::vector<T>(randomSamples);
    }



    /**
     * Resets the number of random samples
     * @param param1 New number of random samples.
     * @param param2 Unused. Attempting to use throws an exception.
     */
    void resetParameters(std::optional<int> param1 = std::nullopt,
			 std::optional<int> param2 = std::nullopt) override {
      if (param2)
	throw std::invalid_argument("The RandomSamplingPermuted class has only "
				    "one parameter, but tried to reset two "
				    "parameters.");
      if (param1)
	setRandomSamples(*param1);
    }



    /**
     * Resets the pseudorandom number generator.
     * @param seed New seed or nullopt for unpredictable initialization.
     */
    inline void resetSeed(std::optional<KdeEstimator::
			  PseudoRandomNumberGenerator::ValueType> seed =
			  std::nullopt) override {
      KdeEstimatorT<T>::resetSeed(seed);
      if (this->X) {
	this->fit(this->n, this->d, this->X);
      }
    }




#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    inline const T* getXpermuted() const {
      return &Xpermuted[0];
    }



    inline int getSampleIdx() const {
      return sampleIdx;
    }



    inline const T* getXSqNorm() const {
      return &XSqNorm[0];
    }
#endif


    
  private:
    void fitImpl() override {
      if (this->n < randomSamples)
	throw std::invalid_argument("Too small dataset fitted: a dataset of " +
				    std::to_string(this->n) + " supplied, but "
				    + std::to_string(randomSamples) + " random "
				    "samples requested.");
      setRandomSamples(randomSamples);
      XSqNorm = std::vector<T>(this->n);
      std::vector<uint32_t> indices(this->n);
      for (int i = 0; i < this->n; ++i)
	indices[i] = i;
      std::mt19937 mt19937rng(this->rngSeed);
      std::shuffle(indices.begin(), indices.end(), mt19937rng);
      Xpermuted = std::vector<T>(this->n * this->d);
      for (int i = 0; i < this->n; ++i)
	array::mov(this->d, this->X + indices[i]*this->d, &Xpermuted[0] +
		   i*this->d);
      array::sqNorm(this->n, this->d, &Xpermuted[0], &XSqNorm[0]);
    }


    
    void queryImpl(const T* q, T* Z, int* samples) const override {
      if (randomSamples + sampleIdx > this->n) {
	int m1 = this->n - sampleIdx;
	int m2 = randomSamples - m1;
	T Z1 = kdeEuclideanMatmul(m1, this->d, this->h,
				  &Xpermuted[0] + sampleIdx*this->d, q,
				  &XSqNorm[0] + sampleIdx, &scratch[0],
				  this->K);
	T Z2 = kdeEuclideanMatmul(m2, this->d, this->h, &Xpermuted[0], q,
				  &XSqNorm[0], &scratch[0], this->K);
	*Z = (Z1*m1 + Z2*m2)/randomSamples;
      }
      else {
	*Z = kdeEuclideanMatmul(randomSamples, this->d, this->h,
                                &Xpermuted[0] + sampleIdx*this->d, q,
                                &XSqNorm[0] + sampleIdx, &scratch[0],
                                this->K);
      }
      sampleIdx = (sampleIdx + randomSamples) % this->n;
      if (samples)
        *samples = randomSamples;
    }



    void queryImpl(int m, const T* Q, T* Z, int* samples) const override {
      for (int i = 0; i < m; ++i) 
        queryImpl(Q + i*this->d, Z + i, samples ? samples + i : nullptr);
    }



    int randomSamples = 0;
    mutable int sampleIdx = 0;
    std::vector<T> XSqNorm;
    mutable std::vector<T> scratch;
    std::vector<T> Xpermuted;
  };
}

#endif // DEANN_RANDOM_SAMPLING_PERMUTED
