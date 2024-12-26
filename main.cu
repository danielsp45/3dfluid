#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <omp.h>

#define SIZE 84

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;
static float *dens_res;

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        std::cerr << "CUDA Error: " << msg << ", " << cudaGetErrorString(err) << ")" << std::endl;
        exit(-1);
    }
}

// Function to allocate simulation data
int allocate_data() {
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

    cudaMalloc((void**) &u, size);
    cudaMalloc((void**) &v, size);
    cudaMalloc((void**) &w, size);
    cudaMalloc((void**) &u_prev, size);
    cudaMalloc((void**) &v_prev, size);
    cudaMalloc((void**) &w_prev, size);
    cudaMalloc((void**) &dens, size);
    cudaMalloc((void**) &dens_prev, size);

    checkCUDAError("Memory allocation failed");

    dens_res = new float[size];
    if (!dens_res) {
        std::cerr << "Memory allocation failed" << std::endl;
        return 0;
    }

    return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);
    
    cudaMemset(u, 0, size);
    cudaMemset(v, 0, size);
    cudaMemset(w, 0, size);
    cudaMemset(u_prev, 0, size);
    cudaMemset(v_prev, 0, size);
    cudaMemset(w_prev, 0, size);
    cudaMemset(dens, 0, size);
    cudaMemset(dens_prev, 0, size);

    checkCUDAError("Clear data failed");
}

// Free allocated memory
void free_data() {
    cudaFree(u);
    cudaFree(v);
    cudaFree(w);
    cudaFree(u_prev);
    cudaFree(v_prev);
    cudaFree(w_prev);
    cudaFree(dens);
    cudaFree(dens_prev);

    checkCUDAError("Free memory failed");

    delete[] dens_res;
}

__global__
void update_dens(float *dens, int idx, float density) {
    dens[idx] = density;
}

__global__
void update_uvw(float *u, float *v, float *w, int idx, float fx, float fy, float fz) {
    u[idx] = fx;
    v[idx] = fy;
    w[idx] = fz;
}

// Apply events (source or force) for the current timestep
void apply_events(const std::vector<Event> &events) {
    int i = M / 2, j = N / 2, k = O / 2;
    int idx = IX(i, j, k);

    bool dens_updated = false;
    float density = 0.0f;

    bool force_updated = false;
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (const auto &event : events) {
        if (event.type == ADD_SOURCE) {
            // Apply density source at the center of the grid
            dens_updated = true;
            density = event.density;
        } else if (event.type == APPLY_FORCE) {
            // Apply forces based on the event's vector (fx, fy, fz)
            force_updated = true;
            fx = event.force.x;
            fy = event.force.y;
            fz = event.force.z;
        }
    }

    // dens and uvw are already on the device (a single thread will update both)
    if (dens_updated) {
        update_dens<<<1, 1>>>(dens, idx, density);
    }

    if (force_updated) {
        update_uvw<<<1, 1>>>(u, v, w, idx, fx, fy, fz);
    }
}

// Function to sum the total density
float sum_density() {
    float total_density = 0.0f;
    int size = (M + 2) * (N + 2) * (O + 2);
    for (int i = 0; i < size; i++) {
        total_density += dens_res[i];
    }
    return total_density;
}

// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
    for (int t = 0; t < timesteps; t++) {
        // Get the events for the current timestep
        std::vector<Event> events = eventManager.get_events_at_timestamp(t);

        // Apply events to the simulation
        apply_events(events);

        // Perform the simulation steps
        vel_step(M, N, O, u, v, w, u_prev, v_prev, w_prev, visc, dt);
        dens_step(M, N, O, dens, dens_prev, u, v, w, diff, dt);
    }

    // Copy the data back to the host
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);
    cudaMemcpy(dens_res, dens, size, cudaMemcpyDeviceToHost);
}

int main() {
    // Initialize EventManager
    EventManager eventManager;
    eventManager.read_events("events.txt");

    // Get the total number of timesteps from the event file
    int timesteps = eventManager.get_total_timesteps();

    // Allocate and clear data
    if (!allocate_data())
        return -1;
    clear_data();

    // Run simulation with events
    simulate(eventManager, timesteps);

    // Print total density at the end of simulation
    float total_density = sum_density();
    std::cout << "Total density after " << timesteps
              << " timesteps: " << total_density << std::endl;

    // Free memory
    free_data();

    return 0;
}