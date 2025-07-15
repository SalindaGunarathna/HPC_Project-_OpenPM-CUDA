2D Heat Diffusion Equation Solvers
This repository contains two implementations of a 2D heat diffusion equation solver using finite difference methods: a serial version and an OpenMP parallelized version. Both solvers demonstrate numerical methods for solving partial differential equations and provide performance benchmarking capabilities.

Overview
The heat diffusion equation (also known as the heat equation) is a parabolic partial differential equation that describes the distribution of heat in a given region over time. The 2D form of the equation is:

text
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
Where:

u(x,y,t) is the temperature at position (x,y) and time t

α is the thermal diffusivity coefficient

The equation describes how heat spreads through a 2D domain

Mathematical Background
Finite Difference Discretization
The continuous partial differential equation is discretized using finite differences:

Spatial derivatives: Central difference scheme

text
∂²u/∂x² ≈ (u[i+1][j] - 2*u[i][j] + u[i-1][j]) / (dx²)
∂²u/∂y² ≈ (u[i][j+1] - 2*u[i][j] + u[i][j-1]) / (dy²)
Temporal derivative: Forward difference (explicit scheme)

text
∂u/∂t ≈ (u_new[i][j] - u[i][j]) / dt
Initial Conditions
Both implementations use a Gaussian heat source as the initial condition:

c
u[i][j] = exp(-50*(x*x + y*y))
This creates a concentrated heat source at the center of the domain that diffuses outward over time.

Implementation Details
Grid Parameters
Grid size: 200×200 points (Nx = Ny = 200)

Domain size: 1.0×1.0 units (Lx = Ly = 1.0)

Time steps: 1000 iterations

Thermal diffusivity: α = 0.0001

Stability Condition
The explicit finite difference scheme requires adherence to the CFL (Courant-Friedrichs-Lewy) condition for numerical stability:

text
dt ≤ 0.25 * min(dx², dy²) / α
This ensures that information propagates no more than one grid cell per time step, maintaining numerical stability.

Code Structure
Serial Implementation (serial_solver.c)
Key Features:

Standard C implementation with <stdio.h>, <stdlib.h>, <math.h>

High-resolution timing using clock_gettime(CLOCK_MONOTONIC, &t)

Dynamic memory allocation for 2D arrays

Simple nested loops for computation

Memory Layout:

c
double **u    = malloc(Nx*sizeof(double*));
double **uNew = malloc(Nx*sizeof(double*));
for(int i=0;i<Nx;i++){
    u[i]    = malloc(Ny*sizeof(double));
    uNew[i] = malloc(Ny*sizeof(double));
}
OpenMP Parallel Implementation (openmp_solver.c)
Key Features:

OpenMP parallelization with #pragma omp parallel for collapse(2)

Automatic thread detection using omp_get_max_threads()

OpenMP timing functions for better accuracy

Parallel initialization and computation loops

Parallelization Strategy:

c
#pragma omp parallel for collapse(2)
for(int i=1;i<Nx-1;i++){
    for(int j=1;j<Ny-1;j++){
        // Computation kernel
    }
}
The collapse(2) directive combines both nested loops into a single parallel loop, enabling better load balancing and utilizing more threads effectively.

Performance Metrics
MLUPS (Million Lattice Updates Per Second)
Both implementations measure performance using MLUPS:

c
double updates = (double)Nt*(Nx-2)*(Ny-2);
double mlups = updates / elapsed / 1e6;
MLUPS indicates how many millions of grid points are updated per second, providing a standardized measure of computational throughput for lattice-based simulations.

Typical Performance Expectations
Serial: 10-50 MLUPS (depending on hardware)

OpenMP: 50-400 MLUPS (scaling with thread count)

Performance scaling: Near-linear speedup expected up to the number of physical cores

Compilation and Usage
Compilation
Serial version:

bash
gcc -O3 -lm serial_solver.c -o serial_solver
OpenMP version:

bash
gcc -O3 -fopenmp -lm openmp_solver.c -o openmp_solver
Execution
Serial:

bash
./serial_solver
OpenMP (controlling thread count):

bash
export OMP_NUM_THREADS=8
./openmp_solver
Output Interpretation
Both programs output:

Time: Wall-clock execution time in seconds

Throughput: Performance in MLUPS

u_center: Final temperature value at the center point (for verification)

Example output:

text
OpenMP run (8 threads):
  Time           : 0.234567 s
  Throughput     : 168.42 MLUPS
  u_center (mid) : 0.012345
Memory Considerations
Memory Usage
Total allocation: 2 × Nx × Ny × sizeof(double) ≈ 640 KB for default parameters

Memory pattern: Row-major layout with pointer-to-pointer structure

Cache efficiency: The nested loop structure provides good spatial locality

Memory Access Pattern
The implementation uses a "ping-pong" approach:

Read from u array

Write to uNew array

Copy back from uNew to u

This avoids race conditions and ensures numerical correctness.

Numerical Properties
Convergence and Accuracy
Order of accuracy: O(dt, dx², dy²) - first-order in time, second-order in space

Conservation: The scheme conserves total energy in the absence of boundary effects

Dissipation: Numerical diffusion may occur, but is minimized by the chosen discretization

Boundary Conditions
Type: Homogeneous Neumann (zero-gradient) boundaries

Implementation: Boundary values remain unchanged throughout simulation

Physical meaning: Insulated boundaries with no heat flux

Optimization Opportunities
Algorithmic Improvements
Implicit schemes: For larger time steps (unconditionally stable)

Multigrid methods: For faster convergence

Adaptive time stepping: For improved efficiency

Implementation Optimizations
Memory layout optimization: Single-dimensional arrays for better cache performance

Vectorization: SIMD instructions for computational kernels

GPU acceleration: CUDA or OpenCL implementations

Parallel Improvements
MPI: For distributed memory systems

Hybrid MPI+OpenMP: For large-scale HPC applications

Task-based parallelism: Using OpenMP tasks for irregular grids

Applications
This solver serves as a foundation for:

Thermal analysis: Heat conduction in materials

Diffusion processes: Chemical species transport

Image processing: Diffusion-based filters

Financial modeling: Option pricing with diffusion terms

References and Further Reading
Finite Difference Methods for Ordinary and Partial Differential Equations

OpenMP Application Programming Interface Specification

Numerical Methods for the Heat Equation

High Performance Computing benchmarking with MLUPS metrics

# Cuda version
## command 
!nvcc -O3 -o cuda_heat cuda_heat.cu
!./cuda_heat