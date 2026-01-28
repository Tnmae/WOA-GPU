#include "utils.cuh"


__device__ void utility::rastrigin_function(float *arr, float &fitness_value) {
  for (int i = 0 ; i < DIM ; i++ ) {
    float xi = arr[i];
    fitness_value += ( xi * xi ) - 10.0f * cosf(2 * CUDART_PI_F * xi) + 10.0f;
  }
}

__device__ void utility::sphere_function(float *arr, float &fitness_value) {
  for (int i = 0 ; i < DIM ; i++ ) {
    float xi = arr[i];
    fitness_value += xi * xi;
  }
}

__global__ void utility::setup_kernel(curandState *dState) {
  int id = threadIdx.x + blockIdx.x + blockDim.x;
  curand_init(clock64() + id * SCALE , id , 0 , &dState[id]);
}

