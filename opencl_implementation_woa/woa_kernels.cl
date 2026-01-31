#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Kernel to update whale positions based on the leader
__kernel void update_whales(
    __global float* positions, 
    __global const float* best_leader,
    const float a, 
    const float r1, 
    const float r2, 
    const int num_whales) 
{
    int i = get_global_id(0);
    if (i >= num_whales) return;

    float A_val = 2.0f * a * r1 - a;
    float C = 2.0f * r2;
    float p = r1; 
    float b = 1.0f; 
    float l = (r2 * 2.0f) - 1.0f; 

    float current_x = positions[i];
    float leader_x = best_leader[0];

    if (p < 0.5f) {
        if (fabs(A_val) < 1.0f) {
            // Encircling Prey
            float D = fabs(C * leader_x - current_x);
            positions[i] = leader_x - A_val * D;
        } else {
            // Search for Prey (Exploration)
            // Using a wider search range for Rastrigin traps
            positions[i] = current_x + A_val * (r1 * 5.12f); 
        }
    } else {
        // Spiral Bubble-net Attack
        float D_prime = fabs(leader_x - current_x);
        positions[i] = D_prime * exp(b * l) * cos(2.0f * M_PI * l) + leader_x;
    }
}

// Updated for Rastrigin Function: f(x) = 10 + (x^2 - 10 * cos(2 * PI * x))
__kernel void calculate_fitness(
    __global const float* pos, 
    __global float* fit,
    const int num_whales) 
{
    int i = get_global_id(0);
    if (i >= num_whales) return;

    float x = pos[i];
    float A = 10.0f;
    fit[i] = A + (x * x - A * cos(2.0f * M_PI * x));
}