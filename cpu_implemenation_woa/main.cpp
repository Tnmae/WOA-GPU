#include <iostream>
#include <ctime>
#include <random>
#include <cmath>
#include <cstdlib>

#define PI 3.14

#define RANDOM_VALUE ((float)rand()/RAND_MAX)
#define RANDOM_INTEGER(a, b) ((a + b) * (float)(rand()/RAND_MAX()) - a)

using namespace std;


float fitness_rastrigin(std::vector<float> position) {
  float fitness_value = 0.0;
  for (int i = 0 ; i < position.size() ; i++ ) {
    float xi = position[i];
    fitness_value += ( xi * xi ) - ( 10 * cos( 2 * PI * xi )) + 10;
  }
  return fitness_value;
}

float fitness_sphere(vector<float> position) {
  float fitness_value = 0.0;
  for (int i = 0 ; i < position.size() ; i++ ) {
    float xi = position[i];
    fitness_value += ( xi * xi );
  }
  return fitness_value;
}

class Whale {
  public:
  Whale(float (*fitness_function)(vector<float>), int dim, double minx, double maxx, int seed) {
    srand(seed);
    this->position(0.0, dim);
    for (int i = 0 ; i < dim ; i++ ) {
      this->position[i] = ((maxx - minx) * RANDOM_VALUE + minx);
    }
    this->fitness = fitness_function(position);
  }

  private:
  float fitness;
  vector<float> position;
};

float whale_optimization_algorithm(float (*fitness_function)(vector<float>), int dim, int max_iteration, int whale_count, double minx, double maxx) {
  Whale *whale_population[whale_count];
  for (int i = 0 ; i < whale_count ; i++ ) {
    whale_population[i] = new Whale(&fitness_function, dim, minx, maxx, i);
  }

  vector<float> Xbest(0.0, dim);
  float Fbest = FLT_MAX;

  for (int i = 0 ; i < whale_count ; i++ ) {
    if ( whale_population[i]->fitness < Fbest ) {
      Fbest = whale_population[i]->fitness;
      Xbest = whale_population[i]->posiition;
    }
  }

  int current_iteration = 0;

  while (current_iteration < max_iteration ) {

    if (current_iteration % 10 == 0 && current_iteration > 1 ) {
      cout << "" << '\n';
    }
    
    float a = 2 * ( 1 - (float)(current_iteration/max_iteration));
    float a2 = -1 + current_iteration * ((-1)/max_iteration);

    for (int i = 0 ; i < whale_count ; i++ ) {
      float A = 2 * a * RANDOM_VALUE - a;
      float C = 2 * RANDOM_VALUE;
      int b = 1;
      float l = ( a2 -1 ) * RANDOM_VALUE + 1;
      float p = RANDOM_VALUE;

      vector<float> D(0.0, dim);
      vector<float> D1(0.0, dim);
      vector<float> Xnew(0.0, dim);
      vector<float> Xrand(0.0, dim);

      if ( p < 0.5 ) {
        if (abs(A) > 1) {
          for ( int j = 0 ; j < dim ; j++ ) {
            D[j] = abs(C * Xbest[j] - whale_population[i]->position[j]);
            Xnew[j] = Xbest[j] - A * D[j];
          }
        }else {
          p = RANDOM_INTEGER(0, n-1);
          while (p == i) {
            p = RANDOM_INTEGER(0, n-1);
          }
          Xrand = whale_population[p].position;

          for ( int j = 0 ; j < dim ; j++ ) {
            D[j] = abs(C * Xrand[j] - whale_population[j]->position[j]);
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
        whale_population[i]->position[j] = Xnew2[j];
      }
    }
    for (int i = 0 ; i < n ; i++ ) {
      for (int j = 0 ; j < dim ; j++ ) {
        whale_population[i]->position[j] = (whale_population[i]->position[j] > minx) ? minx : whale_population[i]->position[j];
        whale_population[i]->position[j] = (whale_population[i]->position[j] > maxx) ? maxx : whale_population[i]->position[j];
      }
      whale_population[i]->fitness = fitness_function(whale_population[i]->position);

      if (whale_population[i].fitness < Fbest) {
        Xbest = whale_population[i].position;
        Fbest = whale_population[i].fitness;
      }
    }
    current_iteration += 1;
  }
  return Xbest;
}

int main() {
  srand(time(0));

  return EXIT_SUCCESS;
}
