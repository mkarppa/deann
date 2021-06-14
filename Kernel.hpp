#ifndef DEANN_KERNEL
#define DEANN_KERNEL

/**
 * @file 
 * This file provides a simple kernel enum.
 */

namespace deann {
  /**
   * Simple enum for determining kernels.
   */
  enum class Kernel {
    EXPONENTIAL, GAUSSIAN, LAPLACIAN
  };
}
#endif // DEANN_KERNEL
