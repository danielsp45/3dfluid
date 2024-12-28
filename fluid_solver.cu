#include "fluid_solver.h"

#include <cstdint>
#include <algorithm>
#include "cuda_utils.h"

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


// Add sources (density or velocity)
__global__
void add_source_kernel(int size, float *x, float *s, float dt) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    const int size = (M + 2) * (N + 2) * (O + 2);
    constexpr int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    CUDA(add_source_kernel<<<blocks, threads_per_block>>>(size, x, s, dt));
}

template<unsigned int boundary_type>
__global__
void set_bnd_kernel(int M, int N, int O, float x_signal, float *x) {
    const uint32_t idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (boundary_type == 0) {
        // Boundary in Z direction
        if (idx1 < M + 1 && idx2 < N + 1) {
            x[IX(idx1, idx2, 0)] = x_signal * x[IX(idx1, idx2, 1)];
            x[IX(idx1, idx2, O + 1)] = x_signal * x[IX(idx1, idx2, O)];
        }
    } else if (boundary_type == 1) {
        // Boundary in Y direction
        if (idx1 < M + 1 && idx2 < O + 1) {
            x[IX(idx1, 0, idx2)] = x_signal * x[IX(idx1, 1, idx2)];
            x[IX(idx1, N + 1, idx2)] = x_signal * x[IX(idx1, N, idx2)];
        }
    } else if (boundary_type == 2) {
        // Boundary in X direction
        if (idx1 < N + 1 && idx2 < O + 1) {
            x[IX(0, idx1, idx2)] = x_signal * x[IX(1, idx1, idx2)];
            x[IX(M + 1, idx1, idx2)] = x_signal * x[IX(M, idx1, idx2)];
        }
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
    dim3 blocksZ((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    CUDA(set_bnd_kernel<0><<<blocksZ, blockDim>>>(M, N, O, x_signal, x));

    // Set y boundaries (M x O)
    dim3 blocksY((M + blockDim.x - 1) / blockDim.x, (O + blockDim.y - 1) / blockDim.y);
    CUDA(set_bnd_kernel<1><<<blocksY, blockDim>>>(M, N, O, x_signal, x));

    // Set x boundaries (N x O)
    dim3 blocksX((N + blockDim.x - 1) / blockDim.x, (O + blockDim.y - 1) / blockDim.y);
    CUDA(set_bnd_kernel<2><<<blocksX, blockDim>>>(M, N, O, x_signal, x));

    // Set corners (1 x 1)
    CUDA(set_bnd_corners_kernel<<<1, 1>>>(M, N, O, x));
}

__global__ void lin_solve_step(
    int M, int N, int O, int b, float *x, const float *x0,
    float cRecip, float cTimesA, int parity, float *max_change) {
    extern __shared__ float sdata[]; // Shared memory for max change reduction
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    float local_max_change = 0.0f;

    if (i <= M && j <= N && k <= O && (i + j + k) % 2 == parity) {
        int idx = IX(i, j, k);
        float old_x = x[idx];
        x[idx] = (x0[idx] * cRecip) +
                 (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                  x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                  x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]) * cTimesA;
        local_max_change = fabsf(x[idx] - old_x);
    }

    // Load local max change into shared memory
    sdata[tid] = local_max_change;
    __syncthreads();

    // Perform block-wide reduction in shared memory
    for (int s = blockDim.x * blockDim.y * blockDim.z / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block's max change to global memory
    if (tid == 0) {
        max_change[blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y] = sdata[0];
    }
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, *d_max_change, *h_max_change;
    const int BLOCK_SIZE = 8;
    int grid_size = (M + BLOCK_SIZE - 1) / BLOCK_SIZE * (N + BLOCK_SIZE - 1) / BLOCK_SIZE * (O + BLOCK_SIZE - 1) /
                    BLOCK_SIZE;

    CUDA(cudaMalloc(&d_max_change, grid_size * sizeof(float)));
    h_max_change = new float[grid_size];

    float cRecip = 1.0f / c;
    float cTimesA = a * cRecip;

    int l = 0;
    do {
        for (int parity = 0; parity < 2; ++parity) {
            dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
            dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (O + BLOCK_SIZE - 1) / BLOCK_SIZE);

            CUDA(lin_solve_step<<<blocks, threads, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(float)>>>(
                     M, N, O, b, x, x0, cRecip, cTimesA, parity, d_max_change));
        }

        // Copy max_change values back and find global max
        CUDA(cudaMemcpy(h_max_change, d_max_change, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
        float max_c = *std::max_element(h_max_change, h_max_change + grid_size);

        // Update boundary conditions (set_bnd equivalent can be implemented here)
        set_bnd(M, N, O, b, x);

        if (max_c < tol) break;
    } while (++l < 20);

    delete[] h_max_change;
    CUDA(cudaFree(d_max_change));
}

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(MAX(M, N), O);
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

__global__
void advect_kernel(float dtX, float dtY, float dtZ, int M, int N, int O, float *d, float *d0, float *u, float *v,
                   float *w) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > M || j > N || k > O) {
        return;
    }

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

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    // #pragma omp parallel for collapse(3)
    //     for (int k = 1; k <= O; k++) {
    //         for (int j = 1; j <= N; j++) {
    //             for (int i = 1; i <= M; i++) {
    //                 float x = i - dtX * u[IX(i, j, k)];
    //                 float y = j - dtY * v[IX(i, j, k)];
    //                 float z = k - dtZ * w[IX(i, j, k)];
    //
    //                 // Clamp to grid boundaries
    //                 x = MAX(0.5f, MIN(M + 0.5f, x));
    //                 y = MAX(0.5f, MIN(N + 0.5f, y));
    //                 z = MAX(0.5f, MIN(O + 0.5f, z));
    //
    //                 int i0 = (int) x, i1 = i0 + 1;
    //                 int j0 = (int) y, j1 = j0 + 1;
    //                 int k0 = (int) z, k1 = k0 + 1;
    //
    //                 float s1 = x - i0, s0 = 1 - s1;
    //                 float t1 = y - j0, t0 = 1 - t1;
    //                 float u1 = z - k0, u0 = 1 - u1;
    //
    //                 d[IX(i, j, k)] =
    //                         s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
    //                               t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
    //                         s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
    //                               t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
    //             }
    //         }
    //     }
    dim3 blockDim(8, 8, 8);
    dim3 blocks((M + blockDim.x - 1) / blockDim.x,
                (N + blockDim.y - 1) / blockDim.y,
                (O + blockDim.z - 1) / blockDim.z);

    CUDA(advect_kernel<<<blocks, blockDim>>>(dtX, dtY, dtZ, M, N, O, d, d0, u, v, w));

    set_bnd(M, N, O, b, d);
}

__global__
void project_kernel_1(int M, int N, int O, float *u, float *v, float *w, float *p, float *div, float halfM) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
        div[IX(i, j, k)] =
                (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
                v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) *
                halfM;
        p[IX(i, j, k)] = 0;
    }
}

__global__
void project_kernel_2(int M, int N, int O, float *u, float *v, float *w, float *p) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
        u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    }
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    const float halfM = -0.5f / MAX(MAX(M, N), O);

    // #pragma omp parallel for collapse(3)
    //     for (int k = 1; k <= O; k++) {
    //         for (int j = 1; j <= N; j++) {
    //             for (int i = 1; i <= M; i++) {
    //                 div[IX(i, j, k)] =
    //                         (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
    //                          v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) *
    //                         halfM;
    //                 p[IX(i, j, k)] = 0;
    //             }
    //         }
    //     }

    dim3 blockDim(8, 8, 8);
    dim3 blocks((M + blockDim.x - 1) / blockDim.x,
                (N + blockDim.y - 1) / blockDim.y,
                (O + blockDim.z - 1) / blockDim.z);
    CUDA(project_kernel_1<<<blocks, blockDim>>>(M, N, O, u, v, w, p, div, halfM));

    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);
    lin_solve(M, N, O, 0, p, div, 1, 6);

    // #pragma omp parallel for collapse(3)
    //     for (int k = 1; k <= O; k++) {
    //         for (int j = 1; j <= N; j++) {
    //             for (int i = 1; i <= M; i++) {
    //                 u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
    //                 v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
    //                 w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    //             }
    //         }
    //     }
    CUDA(project_kernel_2<<<blocks, blockDim>>>(M, N, O, u, v, w, p));

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
