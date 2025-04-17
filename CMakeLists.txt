cmake_minimum_required(VERSION 3.27)
project(cudaSample CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cudaSample main.cu
        functions.h
        MatrixUtils/Matrix.h  )

set_target_properties(cudaSample PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Add optimization flags only for Release builds
target_compile_options(cudaSample PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
        -O3
        --use_fast_math
        >
)
