#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt);

void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt);

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#ifndef LIN_SOLVE_BLOCK_X
#define LIN_SOLVE_BLOCK_X 32
#endif

#ifndef LIN_SOLVE_BLOCK_Y
#define LIN_SOLVE_BLOCK_Y 4
#endif

#ifndef LIN_SOLVE_BLOCK_Z
#define LIN_SOLVE_BLOCK_Z 1
#endif

#define LINEARSOLVERTIMES 20

#endif // FLUID_SOLVER_H
