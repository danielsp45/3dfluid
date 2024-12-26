#include "fluid_solver.h"
#include <cmath>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define LINEARSOLVERTIMES 20
#define BLOCKSIZE 4

#define THREADSPERBLOCK 256

// Add sources (density or velocity)
__global__
void add_source_kernel(int size, float *x, float *s, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    int blocks = (size + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    add_source_kernel<<<blocks, THREADSPERBLOCK>>>(size, x, s, dt);
}

__global__
void set_bnd_z_kernel(int M, int N, int O, float x_signal, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M + 1 && j < N + 1) {
        x[IX(i, j, 0)] = x_signal * x[IX(i, j, 1)];
        x[IX(i, j, O + 1)] = x_signal * x[IX(i, j, O)];
    }
}

__global__
void set_bnd_y_kernel(int M, int N, int O, float x_signal, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M + 1 && k < O + 1) {
        x[IX(i, 0, k)] = x_signal * x[IX(i, 1, k)];
        x[IX(i, N + 1, k)] = x_signal * x[IX(i, N, k)];
    }
}

__global__
void set_bnd_x_kernel(int M, int N, int O, float x_signal, float *x) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < N + 1 && k < O + 1) {
        x[IX(0, j, k)] = x_signal * x[IX(1, j, k)];
        x[IX(M + 1, j, k)] = x_signal * x[IX(M, j, k)];
    }
}

__global__
void set_bnd_corners_kernel(int M, int N, int O, float *x) {
    x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
    x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
    x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
}

// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
    // Precompute x_signal to avoid thread divergence
    float x_signal = (b == 3 || b == 1 || b == 2) ? -1.0f : 1.0f;

    // One kernel per boundary helps reducing thread divergence 
    dim3 blockDim(8, 8);

    // Set z boundaries (M x N)
    dim3 blocksZ(
        (M + blockDim.x - 1) / blockDim.x,
        (N + blockDim.y - 1) / blockDim.y
    );
    set_bnd_z_kernel<<<blocksZ, blockDim>>>(M, N, O, x_signal, x);

    // Set y boundaries (M x O)
    dim3 blocksY(
        (M + blockDim.x - 1) / blockDim.x,
        (O + blockDim.y - 1) / blockDim.y
    );
    set_bnd_y_kernel<<<blocksY, blockDim>>>(M, N, O, x_signal, x);

    // Set x boundaries (N x O)
    dim3 blocksX(
        (N + blockDim.x - 1) / blockDim.x,
        (O + blockDim.y - 1) / blockDim.y
    );
    set_bnd_x_kernel<<<blocksX, blockDim>>>(M, N, O, x_signal, x);

    // Set corners (1 x 1)
    set_bnd_corners_kernel<<<1, 1>>>(M, N, O, x);
}

// Red-black solver with convergence check
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c, old_x, change;
    int l = 0;

    float cRecip = 1.0f / c;
    float cTimesA = a * cRecip;

    do {
        max_c = 0.0f;

        #pragma omp parallel for collapse(2) reduction(max:max_c) private(old_x, change)
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1 + (k + j) % 2; i <= M; i += 2) {
                    old_x = x[IX(i, j, k)];
                    x[IX(i, j, k)] = (x0[IX(i, j, k)] * cRecip) +
                                        (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                           x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                           x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]) * cTimesA;
                    change = fabs(x[IX(i, j, k)] - old_x);
                    if(change > max_c) max_c = change;
                }
            }
        }

        #pragma omp parallel for collapse(2) reduction(max:max_c) private(old_x, change)
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1 + (k + j + 1) % 2; i <= M; i += 2) {
                    old_x = x[IX(i, j, k)];
                    x[IX(i, j, k)] = (x0[IX(i, j, k)] * cRecip) +
                                        (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                           x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                           x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]) * cTimesA;
                    change = fabs(x[IX(i, j, k)] - old_x);
                    if(change > max_c) max_c = change;
                }
            }
        }
        set_bnd(M, N, O, b, x);
    } while (max_c > tol && ++l < 20);
}

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(MAX(M, N), O);
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    #pragma omp parallel for collapse(3)
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                float x = i - dtX * u[IX(i, j, k)];
                float y = j - dtY * v[IX(i, j, k)];
                float z = k - dtZ * w[IX(i, j, k)];

                // Clamp to grid boundaries
                x = MAX(0.5f, MIN(M + 0.5f, x));
                y = MAX(0.5f, MIN(N + 0.5f, y));
                z = MAX(0.5f, MIN(O + 0.5f, z));

                int i0 = (int) x, i1 = i0 + 1;
                int j0 = (int) y, j1 = j0 + 1;
                int k0 = (int) z, k1 = k0 + 1;

                float s1 = x - i0, s0 = 1 - s1;
                float t1 = y - j0, t0 = 1 - t1;
                float u1 = z - k0, u0 = 1 - u1;

                d[IX(i, j, k)] =
                        s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                              t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
                        s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                              t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
            }
        }
    }
    set_bnd(M, N, O, b, d);
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    float halfM = -0.5f / MAX(MAX(M, N), O);

    #pragma omp parallel for collapse(3)
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                div[IX(i, j, k)] =
                        (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
                         v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) *
                        halfM;
                p[IX(i, j, k)] = 0;
            }
        }
    }

    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);
    lin_solve(M, N, O, 0, p, div, 1, 6);

    #pragma omp parallel for collapse(3)
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
                v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
                w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
            }
        }
    }
    set_bnd(M, N, O, 1, u);
    set_bnd(M, N, O, 2, v);
    set_bnd(M, N, O, 3, w);
}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
    add_source(M, N, O, x, x0, dt);
    // SWAP(x0, x);
    diffuse(M, N, O, 0, x0, x, diff, dt);
    // SWAP(x0, x);
    advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
    add_source(M, N, O, u, u0, dt);
    add_source(M, N, O, v, v0, dt);
    add_source(M, N, O, w, w0, dt);
    // SWAP(u0, u);
    diffuse(M, N, O, 1, u0, u, visc, dt);
    // SWAP(v0, v);
    diffuse(M, N, O, 2, v0, v, visc, dt);
    // SWAP(w0, w);
    diffuse(M, N, O, 3, w0, w, visc, dt);
    project(M, N, O, u0, v0, w0, u, v);
    // SWAP(u0, u);
    // SWAP(v0, v);
    // SWAP(w0, w);
    advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
    advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
    advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
    project(M, N, O, u, v, w, u0, v0);
}
