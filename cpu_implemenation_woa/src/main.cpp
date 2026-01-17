#include "../headers/WOA.hpp"

int main() {
  srand(time(0)); //initialize seed once

  std::cout << "Begin whale optimization algorithm on rastrigin function\n";

  int dim = 3;

  std::cout << "Goal is to minimize rastrigin function in " << dim << " variables\nFunction has known minimum = 0.0 at (";
  for (int i = 0 ; i < dim-1 ; i++) {
    std::cout << "0, ";
  }
  std::cout << "0)\n";

  int whale_count = 50;
  int max_iteration = 100;

  std::cout << "Setting number of whales to " << whale_count << '\n';
  std::cout << "Setting maximum iteration to " << max_iteration << '\n';

  std::cout << "Starting Whale Optimization algortihm\n";

  vector<float> best_position = algorithm::whale_optimization_algorithm(utility::fitness_rastrigin, dim, max_iteration, whale_count, -10.0, 10.0);

  std::cout << "Whale optimization algorithm completed\nBest Solution found: ";
  for (int i = 0 ; i < dim ; i++ ) {
    std::cout << best_position[i] << " ";
  }
  
  float error = utility::fitness_rastrigin(best_position);
  std::cout << "\nfitness of best solution= " << error << "\nWhale Optimization Algorithm ended for rastrigin\n"; 

  return EXIT_SUCCESS;
}
