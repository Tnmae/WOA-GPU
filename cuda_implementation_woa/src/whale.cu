#include "whale.cuh"

__device__ Whale::Whale(int *fitness_function, float *minx, float *maxx, curandState *d_state) {
  
  for ( int i = 0; i < DIM ; i++ ) {
    Whale::position[i] = (*maxx - *minx) * curand_uniform(d_state) + *minx;
  }

  switch (*fitness_function) {
    case 1:
      utility::rastrigin_function(Whale::position, Whale::fitness_value);
      break;
    case 2:
      utility::sphere_function(Whale::position, Whale::fitness_value);
      break;
    default:
      errorCode = -1;
      break;
  }
}
