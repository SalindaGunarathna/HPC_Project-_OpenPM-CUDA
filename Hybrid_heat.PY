Hybrid_code = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#define IDX(i,j,N) ((i)*(N)+(j))

__global__ void update_gpu(double *u, double *uNew, int Nx, int Ny, int start, int end, double dx, double dy, double alpha, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= start && i < end && j < Ny-1) {
        double uxx = (u[IDX(i+1,j,Ny)] - 2*u[IDX(i,j,Ny)] + u[IDX(i-1,j,Ny)]) / (dx*dx);
        double uyy = (u[IDX(i,j+1,Ny)] - 2*u[IDX(i,j,Ny)] + u[IDX(i,j-1,Ny)]) / (dy*dy);
        uNew[IDX(i,j,Ny)] = u[IDX(i,j,Ny)] + alpha*dt*(uxx + uyy);
    }
}

int main() {
    int Nx = 200, Ny = 200, Nt = 1000;
    double Lx = 1.0, Ly = 1.0;
    double alpha = 0.0001;
    double dx = Lx / (Nx - 1), dy = Ly / (Ny - 1);
    double dt = 0.25 * fmin(dx*dx, dy*dy) / alpha;
    size_t N = Nx * Ny;

    double *u = (double*)malloc(N * sizeof(double));
    double *uNew = (double*)malloc(N * sizeof(double));

    for(int i = 0; i < Nx; i++) {
        for(int j = 0; j < Ny; j++) {
            double x = i * dx - Lx / 2, y = j * dy - Ly / 2;
            u[IDX(i,j,Ny)] = exp(-50*(x*x + y*y));
        }
    }

    double *d_u, *d_uNew;
    cudaMalloc(&d_u, N * sizeof(double));
    cudaMalloc(&d_uNew, N * sizeof(double));

    cudaMemcpy(d_u, u, N * sizeof(double), cudaMemcpyHostToDevice);

    int cpu_start = 1, cpu_end = Nx/2;
    int gpu_start = Nx/2, gpu_end = Nx - 1;

    dim3 block(16, 16);
    dim3 grid((Nx + block.x - 1)/block.x, (Ny + block.y - 1)/block.y);

    double t0 = omp_get_wtime();

    for (int n = 0; n < Nt; n++) {
        update_gpu<<<grid, block>>>(d_u, d_uNew, Nx, Ny, gpu_start, gpu_end, dx, dy, alpha, dt);
        cudaDeviceSynchronize();

        cudaMemcpy(&uNew[IDX(gpu_start,0,Ny)], &d_uNew[IDX(gpu_start,0,Ny)],
                   (gpu_end - gpu_start) * Ny * sizeof(double), cudaMemcpyDeviceToHost);

        #pragma omp parallel for collapse(2)
        for (int i = cpu_start; i < cpu_end; i++) {
            for (int j = 1; j < Ny - 1; j++) {
                double uxx = (u[IDX(i+1,j,Ny)] - 2*u[IDX(i,j,Ny)] + u[IDX(i-1,j,Ny)]) / (dx*dx);
                double uyy = (u[IDX(i,j+1,Ny)] - 2*u[IDX(i,j,Ny)] + u[IDX(i,j-1,Ny)]) / (dy*dy);
                uNew[IDX(i,j,Ny)] = u[IDX(i,j,Ny)] + alpha*dt*(uxx + uyy);
            }
        }

        double *tmp = u; u = uNew; uNew = tmp;

        cudaMemcpy(d_u, u, N * sizeof(double), cudaMemcpyHostToDevice);
    }

    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;

    double updates = (double)Nt*(Nx-2)*(Ny-2);
    double mlups = updates / elapsed / 1e6;

    printf("Hybrid OpenMP + CUDA run:\\n");
    printf("  Time           : %.6f s\\n", elapsed);
    printf("  Throughput     : %.2f MLUPS\\n", mlups);
    printf("  u_center (mid) : %f\\n", u[IDX(Nx/2, Ny/2, Ny)]);

    free(u);
    free(uNew);
    cudaFree(d_u);
    cudaFree(d_uNew);

    return 0;
}
"""

with open("Hybrid_heat.cu", "w") as f:
    f.write(Hybrid_code)


# !nvcc -Xcompiler -fopenmp -lgomp -o hybrid_run Hybrid_heat.cu
# !./hybrid_run
