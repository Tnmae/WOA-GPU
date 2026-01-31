#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <cmath>

int main() {
    srand(static_cast<unsigned int>(time(0)));

    try {
        // 1. Setup GPU (Intel HD 620)
        cl::Device device = cl::Device::getDefault();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // 2. Load and Build Kernel
        std::ifstream file("woa_kernels.cl");
        if (!file.is_open()) {
            std::cerr << "ERROR: woa_kernels.cl not found!" << std::endl;
            return 1;
        }
        std::string src(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
        cl::Program program(context, src);

        try {
            program.build("-cl-std=CL2.0");
        }
        catch (cl::Error& e) {
            std::cerr << "Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            return 1;
        }

        // 3. Parameters for Rastrigin
        const int num_whales = 100; // Increased for better accuracy
        const int max_iter = 10000;
        std::vector<float> h_pos(num_whales);
        std::vector<float> h_fit(num_whales);

        // Initialize whales between -5.12 and 5.12
        for (int i = 0; i < num_whales; i++) {
            h_pos[i] = ((float)rand() / RAND_MAX) * 10.24f - 5.12f;
        }

        // 4. GPU Buffers
        cl::Buffer d_pos(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_whales, h_pos.data());
        cl::Buffer d_fit(context, CL_MEM_READ_WRITE, sizeof(float) * num_whales);
        cl::Buffer d_best(context, CL_MEM_READ_WRITE, sizeof(float));

        cl::Kernel k_fit(program, "calculate_fitness");
        cl::Kernel k_upd(program, "update_whales");

        float best_x = h_pos[0];
        float global_best_fitness = 1e10;

        std::cout << "Optimizing Rastrigin Function on GPU..." << std::endl;

        // 5. Optimization Loop
        for (int t = 0; t < max_iter; t++) {
            k_fit.setArg(0, d_pos);
            k_fit.setArg(1, d_fit);
            k_fit.setArg(2, num_whales);
            queue.enqueueNDRangeKernel(k_fit, cl::NullRange, cl::NDRange(num_whales));

            queue.enqueueReadBuffer(d_fit, CL_TRUE, 0, sizeof(float) * num_whales, h_fit.data());
            queue.enqueueReadBuffer(d_pos, CL_TRUE, 0, sizeof(float) * num_whales, h_pos.data());

            // Find current leader
            for (int i = 0; i < num_whales; i++) {
                if (h_fit[i] < global_best_fitness) {
                    global_best_fitness = h_fit[i];
                    best_x = h_pos[i];
                }
            }

            // Update Whale Positions
            float a = 2.0f - (t * (2.0f / max_iter));
            queue.enqueueWriteBuffer(d_best, CL_TRUE, 0, sizeof(float), &best_x);

            k_upd.setArg(0, d_pos);
            k_upd.setArg(1, d_best);
            k_upd.setArg(2, a);
            k_upd.setArg(3, (float)rand() / RAND_MAX);
            k_upd.setArg(4, (float)rand() / RAND_MAX);
            k_upd.setArg(5, num_whales);

            queue.enqueueNDRangeKernel(k_upd, cl::NullRange, cl::NDRange(num_whales));

            if (t % 20 == 0) std::cout << "Iteration " << t << " | Best Fitness: " << global_best_fitness << std::endl;
        }

        std::cout << "\n--- Final Result ---" << std::endl;
        std::cout << "Best X: " << best_x << std::endl;
        std::cout << "Global Minimum (Target 0): " << global_best_fitness << std::endl;

    }
    catch (cl::Error& e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
    }

    system("pause");
    return 0;
}