#ifndef WHALE_HPP
#define WHALE_HPP

#include "utils.cuh"

class Whale {
public:
  __device__ Whale(int *fitness_function, float *minx, float *maxx, curandState *dState);

  float fitness_value = 0.0f;
  
  float position[DIM];

  int errorCode = 1;
};

#endif /* WHALE_HPP */
