#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int Nx = 200, Ny = 200;
    int Nt = 1000;
    // Read command line arguments
    if (argc >= 3) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
    }
    if (argc >= 4) {
        Nt = atoi(argv[3]);
    }
    double Lx = 1.0, Ly = 1.0;
    double alpha = 0.0001;
    double dx = Lx/(Nx-1), dy = Ly/(Ny-1);
    double dt = 0.25 * fmin(dx*dx, dy*dy) / alpha;

    // allocate
    double **u    = malloc(Nx*sizeof(double*));
    double **uNew = malloc(Nx*sizeof(double*));
    for(int i=0;i<Nx;i++){
        u[i]    = malloc(Ny*sizeof(double));
        uNew[i] = malloc(Ny*sizeof(double));
    }

    // threads info
    //int threads = omp_get_max_threads();
    int threads = 4;  

    // initial condition
    #pragma omp parallel for collapse(2)
    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny;j++){
        double x = i*dx - Lx/2, y = j*dy - Ly/2;
        u[i][j] = exp(-50*(x*x + y*y));
      }
    }

    // start timer
    double t0 = omp_get_wtime();

    // time‐stepping
    for(int n=0; n<Nt; n++){
      #pragma omp parallel for collapse(2)
      for(int i=1;i<Nx-1;i++){
        for(int j=1;j<Ny-1;j++){
          double uxx = (u[i+1][j] - 2*u[i][j] + u[i-1][j])/(dx*dx);
          double uyy = (u[i][j+1] - 2*u[i][j] + u[i][j-1])/(dy*dy);
          uNew[i][j] = u[i][j] + alpha*dt*(uxx+uyy);
        }
      }
      #pragma omp parallel for collapse(2)
      for(int i=1;i<Nx-1;i++){
        for(int j=1;j<Ny-1;j++){
          u[i][j] = uNew[i][j];
        }
      }
    }

    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;
    FILE *file = fopen("openmp_heat_distribution.csv", "w");
    if (file == NULL) {
        printf("Error: Could not create output file\n");
        return 1;
    }
    // throughput
    double updates = (double)Nt*(Nx-2)*(Ny-2);
    double mlups = updates / elapsed / 1e6;

    // report
    printf("Implementation: OpenMP\n");
    printf("Threads: %d\n", threads);
    printf("GridSize: %dx%d\n", Nx, Ny);
    printf("TimeSteps: %d\n", Nt);
    printf("Time: %.6f\n", elapsed);
    printf("Throughput: %.2f\n", mlups);
    printf("CenterValue: %f\n", u[Nx/2][Ny/2]);
    
    // printf("OpenMP run (%d threads):\n", threads);
    // printf("  Time           : %.6f s\n", elapsed);
    // printf("  Throughput     : %.2f MLUPS\n", mlups);
    // printf("  u_center (mid) : %f\n", u[Nx/2][Ny/2]);

    for(int i = 0; i < Nx; i++) {
        for(int j = 0; j < Ny; j++) {
            fprintf(file, "%.10e", u[i][j]);
            if(j < Ny-1) fprintf(file, ",");
        }
        fprintf(file, "\n");
    }
    // cleanup
    for(int i=0;i<Nx;i++){
      free(u[i]);
      free(uNew[i]);
    }
    free(u);
    free(uNew);
    return 0;
}
