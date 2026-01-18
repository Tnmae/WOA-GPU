#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cfloat>

#define N 100
#define WHALE_COUNT 50
#define DIM 3

using namespace std;

__device__ void rastrigin_function(float *arr, float &fitness_value) {
  for (int i = 0 ; i < DIM ; i++) {
    float xi = arr[i];
    fitness_value += (xi * xi) - 10.0f * (cosf(2 * CUDART_PI_F * xi)) + 10.0f;
  }
}

__device__ void sphere_function(float *arr, float &fitness_value) {
  for (int i = 0 ; i < DIM ; i++ ) {
    float xi = arr[threadIdx.x];
    fitness_value += (xi * xi);
  }
}

class Whale {
public:
  __device__ Whale(int *fitness_function, float *minx, float *maxx, curandState *d_state) {
    for (int i = 0 ; i < DIM ; i++ ) {
      Whale::position[i] = curand_uniform(d_state) * (*maxx - *minx) + *minx;
    }

    switch (*fitness_function) {
      case 1:
        rastrigin_function(Whale::position, Whale::fitness_value);
        break;
      case 2:
        sphere_function(Whale::position, Whale::fitness_value);
        break;
      default:
        error_code = -1;
        break;
    }
  }

  float position[DIM];
  float fitness_value = 0.0f;
  int error_code = 0;
};

__device__ void init_whale(Whale *buffer, int *fitness_function , float *minx, float *maxx, curandState *d_state) {
  new(buffer) Whale(fitness_function, minx, maxx, d_state);
}

__global__ void init_whale_population(Whale *whale, int *fitness_function, float *minx, float *maxx, curandState *d_state) {
  int id = threadIdx.x;
  curand_init(clock64(), id, 0, d_state);
  for (int i = 0 ; i < WHALE_COUNT ; i++ ) {
    init_whale(whale + i, fitness_function, minx, maxx, d_state);
  }
}



int main() {

  Whale *whale_population;
  float *minx;
  float *maxx;
  int *function_ptr;
  curandState *d_state;

  cudaMallocManaged(&minx, sizeof(float));
  cudaMallocManaged(&maxx, sizeof(float));
  cudaMallocManaged(&whale_population, WHALE_COUNT * sizeof(Whale));
  cudaMallocManaged(&function_ptr, sizeof(int));
  cudaMallocManaged(&d_state, sizeof(curandState));

  *minx = -10.0f;
  *maxx = 10.0f;
  *function_ptr = 1;

  init_whale_population<<<1, 1>>>(whale_population, function_ptr, minx, maxx, d_state);

  cudaDeviceSynchronize();

  for (int i = 0 ; i < WHALE_COUNT ; i++ ) {
    std::cout << "whale " << i << " has fitness value = " << whale_population[i].fitness_value << " \n";
  }

  cudaFree(whale_population);
  cudaFree(minx);
  cudaFree(maxx);

  return EXIT_SUCCESS;
}
