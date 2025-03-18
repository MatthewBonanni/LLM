cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_transformer LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (modify based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "")

# Set up FetchContent for JSON and HDF5
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
    hdf5
    GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
    GIT_TAG hdf5_1.14.6
)
FetchContent_MakeAvailable(hdf5)

# Add executable
add_executable(llm 
    src/main.cu
    src/kernels.cu
    src/llm.cu
    src/tokenizer.cu
    src/layer.cu
    src/io.cu
)

# Include directories
target_include_directories(llm PRIVATE src ${hdf5_SOURCE_DIR}/src)

# Set properties for the target
set_target_properties(llm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link correct CUDA runtime library
target_link_libraries(llm
    PRIVATE CUDA::cudart
    PRIVATE nlohmann_json::nlohmann_json
    PRIVATE stdc++
    PRIVATE hdf5-static
    PRIVATE hdf5_hl-static
)

# Install
install(TARGETS llm
    RUNTIME DESTINATION bin
)
