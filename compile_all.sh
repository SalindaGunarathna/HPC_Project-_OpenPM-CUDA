#!/bin/bash

# Serial version
echo "Compiling serial version..."
gcc -O3 -o serial_heat serial_heat.c -lm

# OpenMP version
echo "Compiling OpenMP version..."
gcc -O3 -fopenmp -o openmp_heat omp_heat.c -lm

# CUDA version (only if CUDA is installed)
if command -v nvcc &>/dev/null; then
    echo "Compiling CUDA version..."
    nvcc -O3 -o cuda_heat cuda_heat.cu
else
    echo "nvcc not found - skipping CUDA compilation"
fi

# Hybrid version (only if CUDA is installed)
if command -v nvcc &>/dev/null; then
    echo "Compiling Hybrid version..."
    nvcc -Xcompiler -fopenmp -lgomp -o hybrid_heat Hybrid_heat.cu
else
    echo "nvcc not found - skipping Hybrid compilation"
fi

echo "Compilation complete!"