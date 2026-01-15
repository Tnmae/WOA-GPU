#include "../headers/whale.hpp"

Whale::Whale(float (*fitness_function)(vector<float>), int dim, double minx, double maxx) {
  std::vector<float> position(dim, 0.0);
  Whale::position = position;
  for (int i = 0 ; i < dim ; i++ ) {
    Whale::position[i] = ((maxx - minx) * RANDOM_VALUE + minx);
  }
  Whale::fitness = fitness_function(Whale::position);
}
