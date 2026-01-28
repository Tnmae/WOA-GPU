#ifndef UTILS_HPP
#define UTILS_HPP

#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cfloat>
#include <math_constants.h>
#include <cstdlib>

#define SCALE 1234
#define BLOCKS 5
#define WHALE_COUNT 100
#define DIM 3
#define MAX_ITERATION 100

using namespace std;

namespace utility {
  __device__ void rastrigin_function(float *arr, float &fitness_value);
  
  __device__ void sphere_function(float *arr, float &fitness_value);

  __global__ void setup_kernel(curandState *dState);

};

#endif /* UTILS_HPP */
