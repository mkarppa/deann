#ifndef DEANN_FAST_RNG
#define DEANN_FAST_RNG

#include <random>
#include <type_traits>
#include <cstdint>
#include <stdexcept>

/**
 * @file 
 * This file provides a random number generator that is faster
 * than what standard library offers.
 */

namespace deann {
  /**
   * From M. Thorup. High Speed Hashing for Integers and Strings. 
   * arXiv:1504.06804v9
   *
   * Computes an l-bit hash given a 32 bit integer input.
   *
   * @param x Value to hash
   * @param l Number of hash bits
   * @param a A 64-bit constant
   * @param b A 64-bit constant
   * @return an l-bit hash value
   */
  inline uint32_t hash_to_bits(uint32_t x, uint32_t l, uint64_t a,
				uint64_t b) {
    return (a*x+b) >> (64-l);
  }



  /**
   * From M. Thorup. High Speed Hashing for Integers and Strings. 
   * arXiv:1504.06804v9
   * 
   * Computes a hash in the range \f$[m]\f$ given a 32 bit integer input.
   *
   * @param x Value to hash
   * @param m Maximum value
   * @param a A 64-bit constant
   * @param b A 64-bit constant
   * @return a hash value in the range \f$[m]\f$
   */
  inline uint32_t hash_to_range(uint32_t x, uint32_t m, uint64_t a,
				uint64_t b) {
    return (((a*x+b) >> 32) * m) >> 32;
  }



  /**
   * Computes GCD of a and b
   * @param a left hand operand
   * @param b right hand operand
   * @return GCD(a,b)
   */
  inline uint64_t gcd(uint64_t a, uint64_t b) {
    uint64_t tmp;
    if (b > a)
      std::swap(a,b);
    while (b) {
      tmp = a % b;
      a = b;
      b = tmp;
    }
    return a;
  }

  

  /**
   * A fast random number generator
   */
  class FastRng {
  public:
    typedef uint32_t ValueType;
    
    /**
     * Constructs the object
     * 
     * @param m Number of hash buckets
     * @param seed Random number seed
     */
    explicit FastRng(ValueType m, ValueType seed = std::random_device()()) :
      m(m) {
      if (m == 0)
	throw std::invalid_argument("The parameter m must be positive (got 0)");
      
      std::mt19937 rng(seed);
      auto generateUint64t = [&]() {
	uint64_t tmp = rng();
	return (tmp << 32) | rng();
      };

      do {
	a = generateUint64t();
      }
      while (gcd(a,m) != 1);
      do {
	b = generateUint64t();
      }
      while (gcd(b,m) != 1);

      x = rng();
    }



    /**
     * Returns a random number.
     * 
     * @return A random number in the desired range.
     */
    inline uint32_t operator()() {
      return (((a*x++ + b) >> 32) * m) >> 32;
    }



    /**
     * Fills the array with random numbers.
     * 
     * @param n Number of random values
     * @param y Target array
     */
    inline void operator()(uint64_t n, uint32_t* y) {
      uint32_t X = x;
      for (uint64_t i = 0; i < n; ++i)
      	y[i] = (((a*X++ + b) >> 32) * m) >> 32;
      x = X;
    }

    

#ifndef DEANN_ENABLE_DEBUG_ACCESSORS
  private:
#endif // DEANN_ENABLE_DEBUG_ACCESSORS
    uint32_t m = 0;
    uint64_t a = 0;
    uint64_t b = 0;
    uint32_t x = 0;
  };



  /**
   * Random number generator to provide uniform interface around
   * standard library random number generator (mersenne twister)
   */
  class Mt19937Rng {
  public:
    typedef uint32_t ValueType;


    
    /**
     * Constructs the object
     * 
     * @param m Number of hash buckets
     * @param seed Random number seed
     */
    explicit Mt19937Rng(ValueType m, ValueType seed = std::random_device()()) :
      rng(seed), dist(0,m-1) {
    }


    
    /**
     * Returns a random number.
     * 
     * @return A random number in the desired range.
     */
    ValueType operator()() {
      return dist(rng);
    }

  private:
    std::mt19937 rng;
    std::uniform_int_distribution<ValueType> dist;
  };
}

#endif // DEANN_FAST_RNG
