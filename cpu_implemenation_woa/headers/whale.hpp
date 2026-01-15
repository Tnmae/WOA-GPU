#ifndef WHALE_HPP
#define WHALE_HPP

#include "utils.hpp"

class Whale {
public:
  Whale(float (*fitness_function)(vector<float>), int dim, double minx, double maxx);
  float fitness;
  vector<float> position;
};

#endif /* WHALE_HPP */
