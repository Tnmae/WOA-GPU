#include "../headers/utils.hpp"

float utility::fitness_rastrigin(std::vector<float> position) {
  float fitness_value = 0.0f;
  for (int i = 0 ; i < position.size() ; i++ ) {
    float xi = position[i];
    fitness_value += ( xi * xi ) - ( 10 * cos( 2 * PI * xi )) + 10;
  }
  return fitness_value;
}

float utility::fitness_sphere(vector<float> position) {
  float fitness_value = 0.0;
  for (int i = 0 ; i < position.size() ; i++ ) {
    float xi = position[i];
    fitness_value += ( xi * xi );
  }
  return fitness_value;
}
