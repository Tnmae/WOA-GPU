#include <iostream>
#include <ctime>
#include <random>
#include <cmath>
#include <cstdlib>

#define PI 3.14

#define RANDOM_VALUE ((float)rand()/RAND_MAX)

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
      Xbest = whale_population[i].position;
    }
  }

  int current_iteration = 0;

}

int main() {
  srand(time(0));
  return EXIT_SUCCESS;
}
