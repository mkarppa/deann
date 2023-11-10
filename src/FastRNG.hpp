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

// Enabling this define allows generating 64 bit random numbers (i.e., larger
// datasets for random sampling), implemented using 128-bit arithmetic, but is
// slower
// Note: This option was added *after* the paper was published, so it may have
// negative impact on performance
#define DEANN_128BIT_RNG


namespace deann {
  /**
   * Computes GCD of a and b
   * @param a left hand operand
   * @param b right hand operand
   * @return GCD(a,b)
   */
  template<typename T>
  inline T gcd(T a, T b) {
    T tmp;
    if (b > a)
      std::swap(a,b);
    while (b) {
      tmp = a % b;
      a = b;
      b = tmp;
    }
    return a;
  }


  
  template<typename T>
  int clz(T x);

  

  template<>
  int clz(unsigned long long x) {
    return __builtin_clzll(x);
  }

  

  template<>
  int clz(unsigned long x) {
    return __builtin_clzl(x);
  }

  

  template<>
  int clz(unsigned x) {
    return __builtin_clz(x);
  }
  


  /**
   * A fast random number generator with 128-bit arithmetic
   * This is slower than the 32-bit version
   * 64-bit numbers are generated
   */
  class FastRng128 {
  public:
    typedef uint64_t ValueType;
    typedef uint32_t SeedType;
    static_assert(std::is_same<SeedType,
		  decltype(std::random_device()())>::value);
    
    /**
     * Constructs the object
     * 
     * @param m Number of hash buckets
     * @param seed Random number seed
     */
    explicit FastRng128(ValueType m, SeedType seed = std::random_device()()) :
      m(m) {
      if (m == 0)
	throw std::invalid_argument("The parameter m must be positive (got 0)");
      
      std::mt19937 rng(seed);
      auto generateUint128t = [&]() {
	unsigned __int128 tmp = rng();
	tmp <<= 32;
	tmp |= rng();
	tmp <<= 32;
	tmp |= rng();
	tmp <<= 32;
	tmp |= rng();
	return tmp;
      };

      do {
	a = generateUint128t();
      }
      while (gcd<unsigned __int128>(a,m) != 1);
      do {
	b = generateUint128t();
      }
      while (gcd<unsigned __int128>(b,m) != 1);

      x = rng();
      x <<= 32;
      x |= rng();
    }



    /**
     * Returns a random number.
     * 
     * @return A random number in the desired range.
     */
    inline ValueType operator()() {
      return (((a*x++ + b) >> 64) * m) >> 64;
    }



    /**
     * Fills the array with random numbers.
     * 
     * @param n Number of random values
     * @param y Target array
     */
    inline void operator()(uint64_t n, ValueType* y) {
      uint64_t X = x;
      for (uint64_t i = 0; i < n; ++i)
      	y[i] = (((a*X++ + b) >> 64) * m) >> 64;
      x = X;
    }

    

  private:
    uint64_t m = 0;
    unsigned __int128 a = 0;
    unsigned __int128 b = 0;
    uint64_t x = 0;
  };

  

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
   * A fast random number generator
   */
  class FastRng32 {
  public:
    typedef uint32_t ValueType;
    
    /**
     * Constructs the object
     * 
     * @param m Number of hash buckets
     * @param seed Random number seed
     */
    explicit FastRng32(ValueType m, ValueType seed = std::random_device()()) :
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
      while (gcd<uint64_t>(a,m) != 1);
      do {
	b = generateUint64t();
      }
      while (gcd<uint64_t>(b,m) != 1);

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


#ifdef DEANN_128BIT_RNG
  typedef FastRng128 FastRng;
#else // DEANN_128BIT_RNG
  typedef FastRng32 FastRng;
#endif // DEANN_128BIT_RNG
  
}

#endif // DEANN_FAST_RNG
