#include "WOA.cuh"

__device__ inline void algorithm::init_whale(Whale *whale, int *fitness_function, float *minx, float *maxx, curandState *dState) {
  new (whale) Whale(fitness_function, minx, maxx, dState);
}

__global__ void algorithm::init_whale_population(Whale *whale, int *fitness_function, float *minx, float *maxx, curandState *dState) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int whale_number = threadIdx.x + blockIdx.x * BLOCKS;
  curandState localState = dState[id];
  init_whale(whale + whale_number, fitness_function, minx, maxx, &localState);
}

__global__ void algorithm::whale_optimization_algorithm(Whale *whale_population, int *fitness_function, float *Fbest, float *Xbest, float *minx, float *maxx, curandState *d_state) {
  int current_whale = threadIdx.x + blockIdx.x * BLOCKS;
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = d_state[id];

  if (whale_population[current_whale].fitness_value < *Fbest) {
    *Fbest = whale_population[current_whale].fitness_value;
    for (int i = 0 ; i < DIM ; i++ ) {
      Xbest[i] = whale_population[current_whale].position[i];
    }
  }

  int current_iteration = 0;

  while (current_iteration < MAX_ITERATION) {

    float a = 2.0f * (1.0f - (float)(current_iteration/MAX_ITERATION));
    float a2 = -1.0f + current_iteration * (float)((-1)/MAX_ITERATION);

    float A = 2.0f * a * curand_uniform(&localState) - a;
    float C = 2.0f * curand_uniform(&localState);
    int b = 1;
    float l = (a2 - 1.0f) * curand_uniform(&localState);
    float p = curand_uniform(&localState);

    float D[DIM];
    float D1[DIM];
    float Xnew[DIM];
    float Xrand[DIM];

    if (p < 0.5) {
      if (fabsf(A) > 1 ) {
        for (int j = 0 ; j < DIM; j++ ) {
          D[j] = abs(C * Xbest[j] - whale_population[current_whale].position[j]);
          Xnew[j] = Xbest[j] - A * D[j];
        }
      }else {
        p = (WHALE_COUNT - 1) * curand_uniform(&localState);
        while (p == current_whale) {
          p = (WHALE_COUNT - 1) * curand_uniform(&localState);
        }
        
        for (int j = 0 ; j < DIM ; j++ ) {
          Xrand[j] = whale_population[int(p)].position[j];
        }

        for (int j = 0 ; j < DIM ; j++ ) {
          D[j] = abs(C * Xrand[j] - whale_population[current_whale].position[j]);
          Xnew[j] = Xrand[j] - A * D[j];
        }
      }
    }else {
      for (int j = 0 ; j < DIM ; j++ ) {
        D1[j] = abs(Xbest[j] - whale_population[current_whale].position[j]);
        Xnew[j] = D1[j] * exp(b * l) * cos( 2 * CUDART_PI_F * l ) + Xbest[j];
      }
    }

    for (int j = 0 ; j < DIM ; j++ ) {
      whale_population[current_whale].position[j] = Xnew[j];
    }

    for (int j = 0 ; j < DIM ; j++ ) {
      whale_population[current_whale].position[j] = (whale_population[current_whale].position[j] < *minx) ? *minx : whale_population[current_whale].position[j];
      whale_population[current_whale].position[j] = (whale_population[current_whale].position[j] > *maxx) ? *maxx : whale_population[current_whale].position[j];
    }

    switch (*fitness_function) {
      case 1:
        utility::rastrigin_function(whale_population[current_whale].position, whale_population[current_whale].fitness_value);
        break;
      case 2:
        utility::sphere_function(whale_population[current_whale].position, whale_population[current_whale].fitness_value);
        break;
    }

    if (whale_population[current_whale].fitness_value < *Fbest) {
      Xbest = whale_population[current_whale].position;
      *Fbest = whale_population[current_whale].fitness_value;
    }
    current_iteration++;
  }
}

