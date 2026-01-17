#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <curand.h>

#define RANDOM_FLOAT (((float)rand()/RAND_MAX))
#define RANDOM_FLOAT_RANGE(a, b) ((b - a) * ((float)rand() / RAND_MAX) + a)
#define N 100
#define WHALE_COUNT 50
#define DIM 3

using namespace std;

__device__ void rastrigin_function(float *arr, float *fitness_value) {
  float xi = arr[threadIdx.x];
  float term = (xi * xi) - 10.0f * (cosf(2 * CUDART_PI_F * xi)) + 10.0f;
  atomicAdd(fitness_value, term);
}

__host__ void initVec(float *arr) {
  for (int i = 0 ; i < DIM ; i++ ) {
    arr[i] = rand() % 100;
  }
}

class Whale {
public:
  __host__ __device__ Whale(void (*fitness_function)(float* , float*), float *minx, float *maxx) {
    for (int i = 0 ; i < DIM ; i++ ) {
      Whale::position[i] = RANDOM_FLOAT * (*maxx - *minx) + *minx;
    }
    fitness_function(Whale::position, Whale::fitness_value);
  }

  float position[DIM];
  float *fitness_value;
};

__device__ void init_whale(Whale *buffer, void (*fitness_function)(float *, float *), float *minx, float *maxx) {
  new(buffer + threadIdx.x) Whale(fitness_function, minx, maxx);
}

__global__ void init_whale_population(Whale *whale, void (*fitness_function)(float *, float *), float *minx, float *maxx) {
  init_whale(whale, fitness_function, minx, maxx);
}

int main() {
  srand(time(NULL));

  Whale *whale_population;
  float *minx;
  float *maxx;

  cudaMallocManaged(&minx, sizeof(float));
  cudaMallocManaged(&maxx, sizeof(float));
  cudaMallocManaged(&whale_population, WHALE_COUNT * sizeof(Whale));

  *minx = -10.0f;
  *maxx = 10.0f;

  init_whale_population<<<1, WHALE_COUNT>>>(whale_population, rastrigin_function, minx, maxx);

  cudaDeviceSynchronize();

  for (int i = 0 ; i < WHALE_COUNT ; i++ ) {
    std::cout << *whale_population[i].fitness_value << '\n';
  }

  cudaFree(whale_population);
  cudaFree(minx);
  cudaFree(maxx);

  return EXIT_SUCCESS;
}
