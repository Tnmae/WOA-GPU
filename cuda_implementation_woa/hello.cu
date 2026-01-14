#include <stdio.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error));
        return -1;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);

        if (error != cudaSuccess) {
            printf("Failed to get properties for device %d: %s\n",
                   i, cudaGetErrorString(error));
            continue;
        }

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n",
               prop.major, prop.minor);
        printf("  Total Global Memory: %lu MB\n",
               prop.totalGlobalMem / (1024*1024));
    }

    return 0;
}

// compile with: nvcc -I${CUDA_PATH}/include -L${CUDA_PATH}/lib64 -L${CUDA_PATH}/lib hello.cu -o hello
