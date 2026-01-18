#include <stdio.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

__global__ void generate_random_arrayf(curandState *state, float *arr, float *maxx, float *minx) {
  int id = threadIdx.x;
  curand_init(clock64(), id, 0, state);
  for (int i = 0 ; i < 100 ; i++ ) {
    arr[i] = (*maxx - *minx) * curand_uniform(state) + *minx;
  }
}

int main() {
  curandState *d_state;
  float *arr;
  float *maxx;
  float *minx;

  cudaMallocManaged(&d_state, sizeof(curandState));
  cudaMallocManaged(&arr, 100 * sizeof(float));
  cudaMallocManaged(&maxx, sizeof(float));
  cudaMallocManaged(&minx, sizeof(float));

  *minx = -10.0f;
  *maxx = 10.0f;

  generate_random_arrayf<<<1, 1>>>(d_state, arr, maxx, minx);

  cudaDeviceSynchronize();

  for (int i = 0 ; i < 100 ; i++ ) {
    printf("array[%d] = %.5f\n", i, arr[i]);
  }

  return EXIT_SUCCESS;
}
