#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int Nx = 200, Ny = 200;        // grid points
    int Nt = 1000;                 // time steps
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

    // initial condition
    for(int i=0;i<Nx;i++){
      for(int j=0;j<Ny;j++){
        double x = i*dx - Lx/2, y = j*dy - Ly/2;
        u[i][j] = exp(-50*(x*x + y*y));
      }
    }

    // start timer
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // time‐stepping
    for(int n=0; n<Nt; n++){
      for(int i=1;i<Nx-1;i++){
        for(int j=1;j<Ny-1;j++){
          double uxx = (u[i+1][j] - 2*u[i][j] + u[i-1][j])/(dx*dx);
          double uyy = (u[i][j+1] - 2*u[i][j] + u[i][j-1])/(dy*dy);
          uNew[i][j] = u[i][j] + alpha*dt*(uxx+uyy);
        }
      }
      // swap
      for(int i=1;i<Nx-1;i++)
        for(int j=1;j<Ny-1;j++)
          u[i][j] = uNew[i][j];
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec)
                   + 1e-9*(t1.tv_nsec - t0.tv_nsec);

    // compute throughput
    double updates = (double)Nt*(Nx-2)*(Ny-2);
    double mlups = updates / elapsed / 1e6;

    // report
    printf("Serial run:\n");
    printf("  Time           : %.6f s\n", elapsed);
    printf("  Throughput     : %.2f MLUPS\n", mlups);
    printf("  u_center (mid) : %f\n", u[Nx/2][Ny/2]);

    // cleanup
    for(int i=0;i<Nx;i++){
      free(u[i]);
      free(uNew[i]);
    }
    free(u);
    free(uNew);
    return 0;
}
