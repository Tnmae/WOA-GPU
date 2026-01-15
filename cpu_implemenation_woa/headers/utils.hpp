#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <ctime>
#include <random>
#include <cmath>
#include <cstdlib>
#include <cfloat>

#define PI 3.14

#define RANDOM_VALUE ((float)rand()/RAND_MAX)
#define RANDOM_INTEGER(a, b) (int)((b - a) * (float)rand()/RAND_MAX + a)

using namespace std;

namespace utility {

  float fitness_rastrigin(std::vector<float> position);

  float fitness_sphere(vector<float> position);

}


#endif /* UTILS_HPP */
