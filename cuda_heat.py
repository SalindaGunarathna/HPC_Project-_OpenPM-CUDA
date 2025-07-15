cuda_code = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX(i,j,N) ((i)*(N)+(j))

__global__ void update_kernel(double *u, double *uNew, int Nx, int Ny, double dx, double dy, double alpha, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < Nx-1 && j < Ny-1) {
        double uxx = (u[IDX(i+1,j,Ny)] - 2*u[IDX(i,j,Ny)] + u[IDX(i-1,j,Ny)])/(dx*dx);
        double uyy = (u[IDX(i,j+1,Ny)] - 2*u[IDX(i,j,Ny)] + u[IDX(i,j-1,Ny)])/(dy*dy);
        uNew[IDX(i,j,Ny)] = u[IDX(i,j,Ny)] + alpha*dt*(uxx+uyy);
    }
}

int main(int argc, char *argv[]) {
    int Nx = 200, Ny = 200;
    int Nt = 1000;
    double Lx = 1.0, Ly = 1.0;
    double alpha = 0.0001;
    double dx = Lx/(Nx-1), dy = Ly/(Ny-1);
    double dt = 0.25 * fmin(dx*dx, dy*dy) / alpha;
    size_t N = Nx * Ny;

    // Host allocation
    double *u = (double*)malloc(N * sizeof(double));
    double *uNew = (double*)malloc(N * sizeof(double));

    // Initial condition
    for(int i=0;i<Nx;i++){
        for(int j=0;j<Ny;j++){
            double x = i*dx - Lx/2, y = j*dy - Ly/2;
            u[IDX(i,j,Ny)] = exp(-50*(x*x + y*y));
        }
    }

    // Device allocation
    double *d_u, *d_uNew;
    cudaMalloc(&d_u, N * sizeof(double));
    cudaMalloc(&d_uNew, N * sizeof(double));

    // Copy initial data to device
    cudaMemcpy(d_u, u, N * sizeof(double), cudaMemcpyHostToDevice);

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Kernel launch config
    dim3 block(16, 16);
    dim3 grid((Nx-2+block.x-1)/block.x, (Ny-2+block.y-1)/block.y);

    // Time-stepping
    for(int n=0; n<Nt; n++) {
        update_kernel<<<grid, block>>>(d_u, d_uNew, Nx, Ny, dx, dy, alpha, dt);
        // Swap pointers
        double *tmp = d_u; d_u = d_uNew; d_uNew = tmp;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double elapsed = ms * 1e-3;

    // Copy result back
    cudaMemcpy(u, d_u, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute throughput
    double updates = (double)Nt*(Nx-2)*(Ny-2);
    double mlups = updates / elapsed / 1e6;

    // Report
    printf("CUDA run (GPU):\\n");
    printf("  Time           : %.6f s\\n", elapsed);
    printf("  Throughput     : %.2f MLUPS\\n", mlups);
    printf("  u_center (mid) : %f\\n", u[IDX(Nx/2,Ny/2,Ny)]);

    // Cleanup
    free(u);
    free(uNew);
    cudaFree(d_u);
    cudaFree(d_uNew);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
"""

with open("cuda_heat.cu", "w") as f:
    f.write(cuda_code)
    