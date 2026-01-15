#include "../headers/WOA.hpp"

void algorithm::whale_population_cleanup(Whale *whale_population[], int whale_count) {
  for (int i = 0 ; i < whale_count ; i++ ) {
    delete whale_population[i];
  }
}

vector<float> algorithm::whale_optimization_algorithm(float (*fitness_function)(vector<float>), int dim, int max_iteration, int whale_count, double minx, double maxx) {
  Whale* whale_population[whale_count];
  for (int i = 0 ; i < whale_count ; i++ ) {
    whale_population[i] = new Whale(fitness_function, dim, minx, maxx);
  }

  vector<float> Xbest(dim , 0.0);
  float Fbest = FLT_MAX;

  for (int i = 0 ; i < whale_count ; i++ ) {
    if ( whale_population[i]->fitness < Fbest ) {
      Fbest = whale_population[i]->fitness;
      Xbest = whale_population[i]->position;
    }
  }

  int current_iteration = 0;

  while (current_iteration < max_iteration ) {

    if (current_iteration % 10 == 0 && current_iteration > 1 ) {
      cout << "This is the " << current_iteration << "th Iteration" << '\n';
    }
    
    float a = 2.0f * ( 1.0f - (float)(current_iteration/max_iteration));
    float a2 = -1.0f + current_iteration * ((-1)/max_iteration);


    for (int i = 0 ; i < whale_count ; i++ ) {
      float A = 2.0f * a * RANDOM_VALUE - a;
      float C = 2.0f * RANDOM_VALUE;
      int b = 1.0f;
      float l = ( a2 -1.0f ) * RANDOM_VALUE + 1.0f;
      float p = RANDOM_VALUE;
      
      vector<float> D(dim, 0.0);
      vector<float> D1(dim, 0.0);
      vector<float> Xnew(dim, 0.0);
      vector<float> Xrand(dim, 0.0);

      if ( p < 0.5 ) {
        if (abs(A) > 1) {
          for ( int j = 0 ; j < dim ; j++ ) {
            D[j] = abs(C * Xbest[j] - whale_population[i]->position[j]);
            Xnew[j] = Xbest[j] - A * D[j];
          }
        }else {
          p = RANDOM_INTEGER(0, whale_count-1);
          while (p == i) {
            p = RANDOM_INTEGER(0, whale_count-1);
          }
          Xrand = whale_population[(int)p]->position;

          for ( int j = 0 ; j < dim ; j++ ) {
            D[j] = abs(C * Xrand[j] - whale_population[i]->position[j]);
            Xnew[j] = Xrand[j] - A * D[j];
          }
        }
      }else {
        for (int j = 0 ; j < dim ; j++ ) {
          D1[j] = abs(Xbest[j] - whale_population[i]->position[j]);
          Xnew[j] = D1[j] * exp(b * l) * cos( 2 * PI * l ) + Xbest[j];
        }
      }
      for (int j = 0 ; j < dim ; j++ ) {
        whale_population[i]->position[j] = Xnew[j];
      }
    }
    for (int i = 0 ; i < whale_count ; i++ ) {
      for (int j = 0 ; j < dim ; j++ ) {
        whale_population[i]->position[j] = (whale_population[i]->position[j] < minx) ? minx : whale_population[i]->position[j];
        whale_population[i]->position[j] = (whale_population[i]->position[j] > maxx) ? maxx : whale_population[i]->position[j];
      }
      whale_population[i]->fitness = fitness_function(whale_population[i]->position);

      if (whale_population[i]->fitness < Fbest) {
        Xbest = whale_population[i]->position;
        Fbest = whale_population[i]->fitness;
      }
    }
    current_iteration++;
  }

  algorithm::whale_population_cleanup(whale_population , whale_count);

  return Xbest;
}

