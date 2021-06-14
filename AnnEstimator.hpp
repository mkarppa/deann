#ifndef DEANN_ANN_ESTIMATOR
#define DEANN_ANN_ESTIMATOR

#include "RandomSampling.hpp"
#include "FastRNG.hpp"
#include "KdeEstimator.hpp"
#include "Kernel.hpp"
#include "Array.hpp"
#include <unordered_set>
#include <random>
#include <iostream>

/**
 * @file
 * This file implements the DEANN estimator.
 */

namespace deann {
  /**
   * This class represents an estimator that takes in as a template argument an
   * Approximate Nearest Neighbor (ANN) class, uses it to select the near 
   * neighbors of query points, and complements the KDE estimate with random 
   * sampling.
   * 
   * @tparam T Type of data to process, either float or double.
   * @tparam AnnClass A class that implements functions query() and fit()
   * for approximate nearest neighbor detection, see LinearScan for the 
   * interface.
   */
  template<typename T, typename AnnClass>
  class AnnEstimatorBase : public KdeEstimatorT<T> {
  public:
    /**
     * Construct the estimator object. Note: random samples is not provided here
     * because it is processed in a subclass specific manner.
     * 
     * @param bandwidth Desired bandwidth
     * @param kernel The kernel to use.
     * @param nearNeighbors The number \f$k\geq 0\f$ of near neighbors to 
     * consider.
     * @param annEstimator Pointer to the estimator object.
     * @param seed Optional random number seed.
     */
    AnnEstimatorBase(double bandwidth, Kernel kernel, int nearNeighbors,
                     AnnClass* annEstimator, std::optional<KdeEstimator::
                     PseudoRandomNumberGenerator::ValueType> seed =
                     std::nullopt) :
      KdeEstimatorT<T>(bandwidth, kernel, seed), annEstimator(annEstimator) {
      setNearNeighbors(nearNeighbors);
    }



    /**
     * Reset the near neighbors parameter without having to
     * reconstruct the object
     * @param newNeighbors New parameter value
     */
    virtual void setNearNeighbors(int newNeighbors) {
      if (newNeighbors < 0)
	throw std::invalid_argument("Must have a non-negative number of near "
                                    "neighbors");
      nearNeighbors = newNeighbors;
      nnsVector = std::vector<int>(nearNeighbors);
      nnsDistVector = std::vector<T>(nearNeighbors);
      scratch = std::vector<T>(this->d + nearNeighbors + randomSamples);
    }


    
    /**
     * Reset the random samples parameter without having to
     * reconstruct the object
     * @param newSamples New parameter value
     */
    virtual void setRandomSamples(int newSamples) {
      if (newSamples < 0)
	throw std::invalid_argument("Must have a non-negative number of random "
                                    "samples");
      randomSamples = newSamples;
      scratch = std::vector<T>(this->d + nearNeighbors + randomSamples);
    }

    

    /**
     * Reset parameters of the estimator without reconstructing the object.
     * @param param1 The number of near neighbors.
     * @param param2 The number of random samples.
     */
    void resetParameters(std::optional<int> param1 = std::nullopt,
			 std::optional<int> param2 = std::nullopt) override {
      if (param1 && param2) {
        setNearNeighbors(0);
        setRandomSamples(0);
        setNearNeighbors(*param1);
        setRandomSamples(*param2);
      }
      else if (param1) {
	setNearNeighbors(*param1);
      }
      else  if (param2) {
	setRandomSamples(*param2);
      }
    }


  protected:
    void fitImpl() override {
      scratch = std::vector<T>(this->d + nearNeighbors + randomSamples);
    }


    
    int nearNeighbors = 0;
    int randomSamples = 0;
    mutable std::vector<T> scratch;
    

    
  private:
    void queryImpl(const T* q, T* Z, int* samples) const override {
      if (nearNeighbors > 0) {
	annEstimator->query(this->d, nearNeighbors, q, &nnsVector[0],
			    &nnsDistVector[0], samples);
      }
      else if (samples) {
        *samples = 0;
      }
      int k = 0;
      while (k < nearNeighbors && nnsVector[k] >= 0)
	++k;
      
      std::unordered_set<int> nns(nnsVector.begin(), nnsVector.begin() + k);
      
      T Z1 = 0;
      if (k > 0) {
	if (nnsDistVector[0] < 0) {
	  // perform full distance recomputation if no distances are reported
	  Z1 = kdeSubset(k, this->d, this->h, this->X, q, &nnsVector[0],
                         &scratch[0], this->K);
	}
	else {
	  // otherwise, use the precomputed distances
	  Z1 = kdeDists(k, this->h, &nnsDistVector[0], &scratch[0], this->K);
	}
      }
      if (k < this->n && randomSamples > 0) {
        int S;
	T Z2 = randomSamplingImpl(q, nns, &S);
	*Z = Z1 * k / this->n +
	  Z2 * (this->n - k) / this->n;
        if (samples && *samples >= 0)
          *samples += S;
      }
      else {
	*Z = Z1 * k / this->n;
      }
    }


    
    void queryImpl(int m, const T* Q, T* Z, int* samples) const override {
      for (int i = 0; i < m; ++i)
	queryImpl(Q + i*this->d, Z + i, samples ? samples + i : nullptr);
    }



    virtual T randomSamplingImpl(const T* q, const std::unordered_set<int>& nns, int* samples)
      const = 0;


    
    AnnClass* annEstimator = nullptr;
    mutable std::vector<int> nnsVector;
    mutable std::vector<T> nnsDistVector;
  };



  /**
   * This is a variant of the Approximate Nearest Neighbor (ANN) estimator that 
   * uses ``pure'' random sampling as a subroutine.
   * 
   * @tparam T Type of data to be processed (float or dpuble)
   * @tparam AnnClass A class that implements functions query() and fit()
   * for approximate nearest neighbor detection, see LinearScan for the 
   * interface.
   */
  template<typename T, typename AnnClass>
  class AnnEstimator : public AnnEstimatorBase<T,AnnClass> {
  public:
    /**
     * Construct the estimator object.
     * 
     * @param bandwidth Desired bandwidth
     * @param kernel The kernel to use.
     * @param nearNeighbors The number \f$k\geq 0\f$ of near neighbors to 
     * consider.
     * @param randomSamples The number \f$m\geq 0\f$ of random samples to use
     * in the complementory distribution. Note that the estimator is unbiased
     * only if \f$m>0\f$.
     * @param annEstimator Pointer to the estimator object
     * @param seed Optional random number generator seed
     */
    AnnEstimator(double bandwidth, Kernel kernel, int nearNeighbors,
                 int randomSamples, AnnClass* annEstimator,
                 std::optional<KdeEstimator::PseudoRandomNumberGenerator::
                 ValueType> seed = std::nullopt) :
      AnnEstimatorBase<T,AnnClass>(bandwidth, kernel, nearNeighbors,
                                   annEstimator, seed) {
      setRandomSamples(randomSamples);
    }


    /**
     * Reset the random samples parameter without having to
     * reconstruct the object
     * @param newSamples New parameter value
     */
    void setRandomSamples(int newSamples) override {
      AnnEstimatorBase<T,AnnClass>::setRandomSamples(newSamples);
      sampleIdx = std::vector<uint32_t>(this->randomSamples);
    }



  private:
    T randomSamplingImpl(const T* q, const std::unordered_set<int>& nns, int* samples) const
      override {
      this->rng(this->randomSamples, &sampleIdx[0]);
      for (int i = 0; i < this->randomSamples; ++i) {
        uint32_t idx = sampleIdx[i];
        while (nns.count(idx))
          idx = sampleIdx[i] = this->rng();
      }

      *samples = this->randomSamples;

      return kdeSubset(this->randomSamples, this->d, this->h, this->X, q,
                       &sampleIdx[0], &this->scratch[0], this->K);
    }

    
    mutable std::vector<uint32_t> sampleIdx;
  };



  /**
   * This is a variant of the Approximate Nearest Neighbor (ANN)
   * estimator that uses ``permuted'' random sampling as a
   * subroutine. That is, instead of drawing truly independent
   * samples, the dataset is permuted upon fitting, and a consecutive
   * set of rows is used when computing distances to sampled data
   * points. Consequentially, the total number of random samples may
   * vary because duplicates that occur within the near neighbor
   * results are removed.
   * 
   * @tparam T Type of data to be processed (float or dpuble)
   * @tparam AnnClass A class that implements functions query() and fit()
   * for approximate nearest neighbor detection, see LinearScan for the 
   * interface.
   */
  template<typename T, typename AnnClass>
  class AnnEstimatorPermuted : public AnnEstimatorBase<T,AnnClass> {
  public:
    /**
     * Construct the estimator object.
     * 
     * @param bandwidth Desired bandwidth
     * @param kernel The kernel to use.
     * @param nearNeighbors The number \f$k\geq 0\f$ of near neighbors to 
     * consider.
     * @param randomSamples The number \f$m\geq 0\f$ of random samples to use
     * in the complementory distribution. Note that the estimator is unbiased
     * only if \f$m>0\f$.
     * @param annEstimator Pointer to the estimator object
     * @param seed Optional random number generator seed
     */
    AnnEstimatorPermuted(double bandwidth, Kernel kernel, int nearNeighbors,
                         int randomSamples, AnnClass* annEstimator,
                         std::optional<KdeEstimator::
                         PseudoRandomNumberGenerator::ValueType> seed =
                         std::nullopt) :
      AnnEstimatorBase<T,AnnClass>(bandwidth, kernel, nearNeighbors,
                                   annEstimator, seed) {
      if (kernel != Kernel::EXPONENTIAL && kernel != Kernel::GAUSSIAN)
	throw std::invalid_argument("Only exponential and gaussian kernels are "
				    "supported by this class.");
      setRandomSamples(randomSamples);
    }

    

    /**
     * Reset the random samples parameter without having to
     * reconstruct the object
     * @param newSamples New parameter value
     */
    void setRandomSamples(int newSamples) override {
      if (this->X && newSamples + this->nearNeighbors > this->n)
        throw std::invalid_argument("Tried to set too many near neighbors and "
                                    "random samples (" +
                                    std::to_string(this->nearNeighbors) + " + "
                                    + std::to_string(newSamples) + "): this "
                                    "exceeds the number of vectors in the "
                                    "fitted dataset (" + std::to_string(this->n)
                                    + ")");
      AnnEstimatorBase<T,AnnClass>::setRandomSamples(newSamples);
    }

    

    /**
     * Reset the near neighbors parameter without having to reconstruct the object.
     * @param newNeighbors
     */
    void setNearNeighbors(int newNeighbors) override {
      if (this->X && newNeighbors + this->randomSamples > this->n)
        throw std::invalid_argument("Tried to set too many near neighbors and "
                                    "random samples (" +
                                    std::to_string(newNeighbors) + " + "
                                    + std::to_string(this->randomSamples) +
                                    "): this exceeds the number of vectors in "
                                    "the fitted dataset (" +
                                    std::to_string(this->n) + ")");
      AnnEstimatorBase<T,AnnClass>::setNearNeighbors(newNeighbors);
    }

    

#ifdef DEANN_ENABLE_DEBUG_ACCESSORS
    inline const T* getXpermuted() const {
      return &Xpermuted[0];
    }



    inline const uint32_t* getXpermutedIdx() const {
      return &XpermutedIdx[0];
    }



    inline int getSampleIdx() const {
      return sampleIdx;
    }
#endif // DEANN_ENABLE_DEBUG_ACCESSORS



    /**
     * Reset the random number seed.
     * @param seed New seed, or alternatively nullopt for unpredictable default 
     * initialization.
     */
    void resetSeed(std::optional<KdeEstimator::
                   PseudoRandomNumberGenerator::ValueType> seed =
                   std::nullopt) override {
      KdeEstimatorT<T>::resetSeed(seed);
      if (this->X) {
	this->fit(this->n, this->d, this->X);
      }
    }


    
  private:
    T randomSamplingImpl(const T* q, const std::unordered_set<int>& nns, int* samples) const
      override {
      int correctionCount = 0;
      T correctionAmount = 0;
      int k = 0;
      int j = sampleIdx;
      while (k < this->randomSamples) {
        if (nns.count(XpermutedIdx[j])) {
          const T* x = &Xpermuted[0] + j*this->d;
          T dist = array::euclideanDistance(this->d, q, x, &this->scratch[0]);
          correctionAmount += this->K == Kernel::EXPONENTIAL ?
            std::exp(-dist/this->h) : std::exp(-dist*dist/this->h/this->h/2);
          ++correctionCount;
        }
        else {
          ++k;
        }
        j = (j+1) % this->n;
      }
      int m = this->randomSamples + correctionCount;
      
      T Z;
      int m1 = 0;
      int m2 = 0;
      if (m + sampleIdx > this->n) {
        m1 = this->n - sampleIdx;
	m2 = m - m1;
	T Z1 = kdeEuclideanMatmul(m1, this->d, this->h,
				  &Xpermuted[0] + sampleIdx*this->d, q,
				  &XSqNorm[0] + sampleIdx, &this->scratch[0],
				  this->K);
	T Z2 = kdeEuclideanMatmul(m2, this->d, this->h, &Xpermuted[0], q,
				  &XSqNorm[0], &this->scratch[0], this->K);
        Z = (Z1*m1 + Z2*m2)/m;
      }
      else {
        Z = kdeEuclideanMatmul(m, this->d, this->h,
                               &Xpermuted[0] + sampleIdx*this->d, q,
                               &XSqNorm[0] + sampleIdx, &this->scratch[0],
                               this->K);
      }

      if (correctionCount > 0)
        Z = (Z*m - correctionAmount) / this->randomSamples;
      sampleIdx = (sampleIdx + m) % this->n;
      *samples = m;
      return Z;
    }
      


    void fitImpl() override {
      AnnEstimatorBase<T,AnnClass>::fitImpl();
      if (this->randomSamples + this->nearNeighbors > this->n)
        throw std::invalid_argument("Too many random samples and near "
                                    "neighbors requested: the amount exceeds "
                                    "the total number of datapoints");
      
      XSqNorm = std::vector<T>(this->n);
      XpermutedIdx = std::vector<uint32_t>(this->n);
      for (int i = 0; i < this->n; ++i)
	XpermutedIdx[i] = i;
      std::mt19937 mt19937rng(this->rngSeed);
      std::shuffle(XpermutedIdx.begin(), XpermutedIdx.end(), mt19937rng);
      Xpermuted = std::vector<T>(this->n * this->d);
      for (int i = 0; i < this->n; ++i)
	array::mov(this->d, this->X + XpermutedIdx[i]*this->d, &Xpermuted[0] +
		   i*this->d);
      array::sqNorm(this->n, this->d, &Xpermuted[0], &XSqNorm[0]);
    }
    
    std::vector<T> XSqNorm;
    std::vector<T> Xpermuted;
    std::vector<uint32_t> XpermutedIdx;
    mutable int sampleIdx = 0;
  };
}

#endif // DEANN_ANN_ESTIMATOR
