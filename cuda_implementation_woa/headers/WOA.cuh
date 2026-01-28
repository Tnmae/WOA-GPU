#ifndef META_ALGORITHM_HPP
#define META_ALGORITHM_HPP

#include "whale.cuh"

namespace algorithm {
  __device__ inline void init_whale(Whale *whale, int *fitness_function, float *minx, float *maxx, curandState *dState);

  __global__ void init_whale_population(Whale *whale, int *fitness_function, float *minx, float *maxx, curandState *dState);

  __global__ void whale_optimization_algorithm(Whale *whale_population, int *fitness_function, float *Fbest, float *Xbest, float *minx, float *maxx, curandState *d_state);

};

#endif /* META_ALGORITHM_HPP */
