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
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CUDA(add_source_kernel<<<blocks, THREADS_PER_BLOCK>>>(size, x, s, dt));
}

__global__
void set_bnd_x_kernel(int M, int N, int O, float *x, float x_signal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i <= M && j <= N) {
        x[IX(0, i, j)] = x_signal * x[IX(1, i, j)];
        x[IX(M + 1, i, j)] = x_signal * x[IX(M, i, j)];
    }
}

__global__
void set_bnd_y_kernel(int M, int N, int O, float *x, float x_signal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i <= M && j <= O) {
        x[IX(i, 0, j)] = x_signal * x[IX(i, 1, j)];
        x[IX(i, N + 1, j)] = x_signal * x[IX(i, N, j)];
    }
}

__global__
void set_bnd_z_kernel(int M, int N, int O, float *x, float x_signal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i <= N && j <= O) {
        x[IX(i, j, 0)] = x_signal * x[IX(i, j, 1)];
        x[IX(i, j, M + 1)] = x_signal * x[IX(i, j, M)];
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
    dim3 blockDim(THREADS_PER_BLOCK, 1);

    // Set z boundaries (M x N)
    dim3 blocksZ(
        (M + blockDim.x - 1) / blockDim.x,
        (N + blockDim.y - 1) / blockDim.y
    );
    CUDA(set_bnd_x_kernel<<<blocksZ, blockDim>>>(M, N, O, x, x_signal));

    // Set y boundaries (M x O)
    dim3 blocksY(
        (M + blockDim.x - 1) / blockDim.x,
        (O + blockDim.y - 1) / blockDim.y
    );
    CUDA(set_bnd_y_kernel<<<blocksY, blockDim>>>(M, N, O, x, x_signal));

    // Set x boundaries (N x O)
    dim3 blocksX(
        (N + blockDim.x - 1) / blockDim.x,
        (O + blockDim.y - 1) / blockDim.y
    );
    CUDA(set_bnd_z_kernel<<<blocksX, blockDim>>>(M, N, O, x, x_signal));

    // Set corners (1 x 1)
    CUDA(set_bnd_corners_kernel<<<1, 1>>>(M, N, O, x));
}

__global__
void reduce_block_max(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 8) + tid;

    sdata[tid] = 0.0f;

    // First comparison during global load
    for (int i = 0; i < 8; i++) {
        if (idx < n) {
            sdata[tid] = fmaxf(sdata[tid], input[idx]);
        }

        // Strided access to improve memory coalescing
        // avoiding overlaps between threads
        idx += blockDim.x;
    }

    // After this sync point, all threads (total of blockDim.x)
    // will have their own max of 8 elements
    __syncthreads();

    // Reversed-tree reduction (sequential addressing)
    // starting with half of the threads (active threads)
    // we combine the results of the first half with the idle threads
    // until we have only one thread left
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // sdata[tid + s] corresponds to the idle thread
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

float reduce_global_max(float *d_max_changes, float *d_partials, int size) {
    int threads_per_block = 256;

    // We will process 8 elements per thread
    // Effectively reducing kernel launching overhead
    int blocks = (size + (threads_per_block * 8) - 1) / (threads_per_block * 8);

    reduce_block_max<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(d_max_changes, d_partials, size);

    while (blocks > 1) {
        // The final size of the array will be 1 (one global max)
        size = blocks;
        blocks = (blocks + (threads_per_block * 8) - 1) / (threads_per_block * 8);

        // After completion d_partials[k] (for k in [0..blocks - 1])
        // will contain the max per block
        reduce_block_max<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(d_partials, d_partials, size);
    }

    // Global max is stored at d_partials[0]
    float max_c;
    CUDA(cudaMemcpy(&max_c, d_partials, sizeof(float), cudaMemcpyDeviceToHost));

    return max_c;
}

__global__ void lin_solve_kernel(
    int M, int N, int O,
    float *x, const float *x0, float cRecip, float cTimesA,
    int color, float *d_max_changes
) {
    // +1 evicts the need for extra boundary checks, reducing divergence
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    // By using this full indexing, we can avoid the need for extra boundary checks
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1 + (j + k + color) % 2;

    if (i <= M && j <= N && k <= O) {
        int idx = IX(i, j, k);

        float old_x = x[idx];
        x[idx] = (x0[idx] * cRecip) +
                 (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                  x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                  x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]) * cTimesA;

        d_max_changes[idx] = fabsf(x[idx] - old_x);
    }
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c;

    float cRecip = 1.0f / c;
    float cTimesA = a * cRecip;

    dim3 blockDim(LIN_SOLVE_BLOCK_X, LIN_SOLVE_BLOCK_Y, LIN_SOLVE_BLOCK_Z);
    dim3 blocks(
        // M is halved because we will be jumping 2 elements at a time during the kernel
        ((M / 2) + blockDim.x - 1) / blockDim.x,
        (N + blockDim.y - 1) / blockDim.y,
        (O + blockDim.z - 1) / blockDim.z
    );

    int size = (M + 2) * (N + 2) * (O + 2);
    extern float *d_max_changes, *d_partials;

    int l = 0;
    do {
        CUDA(cudaMemset(d_max_changes, 0, size * sizeof(float)));

        // Red & Black
        CUDA(lin_solve_kernel<<<blocks, blockDim>>>(M, N, O, x, x0, cRecip, cTimesA, 0, d_max_changes));
        CUDA(lin_solve_kernel<<<blocks, blockDim>>>(M, N, O, x, x0, cRecip, cTimesA, 1, d_max_changes));

        max_c = reduce_global_max(d_max_changes, d_partials, size);

        set_bnd(M, N, O, b, x);

        if (max_c < tol) break;
    } while (++l < LINEARSOLVERTIMES);
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

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

    dim3 blockDim(LIN_SOLVE_BLOCK_X, LIN_SOLVE_BLOCK_Y, LIN_SOLVE_BLOCK_Z);
    dim3 blocks(
        (M + blockDim.x - 1) / blockDim.x,
        (N + blockDim.y - 1) / blockDim.y,
        (O + blockDim.z - 1) / blockDim.z
    );

    CUDA(advect_kernel<<<blocks, blockDim>>>(dtX, dtY, dtZ, M, N, O, d, d0, u, v, w));

    set_bnd(M, N, O, b, d);
}

__global__
void project_kernel_1(int M, int N, int O, float *u, float *v, float *w, float *p, float *div, float halfM) {
    // +1 evicts the need for extra boundary checks, reducing divergence
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        div[IX(i, j, k)] =
                (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
                 v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) *
                halfM;
        p[IX(i, j, k)] = 0;
    }
}

__global__
void project_kernel_2(int M, int N, int O, float *u, float *v, float *w, float *p) {
    // +1 evicts the need for extra boundary checks, reducing divergence
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    }
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    const float halfM = -0.5f / MAX(MAX(M, N), O);

    dim3 blockDim(LIN_SOLVE_BLOCK_X, LIN_SOLVE_BLOCK_Y, LIN_SOLVE_BLOCK_Z);
    dim3 blocks(
        (M + blockDim.x - 1) / blockDim.x,
        (N + blockDim.y - 1) / blockDim.y,
        (O + blockDim.z - 1) / blockDim.z
    );

    CUDA(project_kernel_1<<<blocks, blockDim>>>(M, N, O, u, v, w, p, div, halfM));

    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);
    lin_solve(M, N, O, 0, p, div, 1, 6);

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
