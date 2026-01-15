#ifndef WAO_HPP
#define WAO_HPP

#include "whale.hpp"

namespace algorithm {
  void whale_population_cleanup(Whale *whale_population[], int whale_count);
  
  vector<float> whale_optimization_algorithm(float (*fitness_function)(vector<float>), int dim, int max_iteration, int whale_count, double minx, double maxx);

};

#endif /* WAO_HPP */
