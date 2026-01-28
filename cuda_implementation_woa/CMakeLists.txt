cmake_minimum_required(VERSION 3.18)

project(WOA_CUDA LANGUAGES CXX CUDA)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable separable compilation (important for multiple .cu files)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find all CUDA source files
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)

# Create executable
add_executable(${CMAKE_PROJECT_NAME}
    ${CUDA_SOURCES}
)

# Include headers
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE headers)

# Set GPU architecture (change if needed)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES native
)

# Optional: fast math + optimization
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
