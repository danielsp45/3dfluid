#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include "cuda_utils.h"

#define SIZE 168

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f; // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;
static float *dens_res;

// Function to allocate simulation data
int allocate_data() {
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

    CUDA(cudaMalloc((void**) &u, size));
    CUDA(cudaMalloc((void**) &v, size));
    CUDA(cudaMalloc((void**) &w, size));
    CUDA(cudaMalloc((void**) &u_prev, size));
    CUDA(cudaMalloc((void**) &v_prev, size));
    CUDA(cudaMalloc((void**) &w_prev, size));
    CUDA(cudaMalloc((void**) &dens, size));
    CUDA(cudaMalloc((void**) &dens_prev, size));

    dens_res = new float[size];

    return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
    int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

    CUDA(cudaMemset(u, 0, size));
    CUDA(cudaMemset(v, 0, size));
    CUDA(cudaMemset(w, 0, size));
    CUDA(cudaMemset(u_prev, 0, size));
    CUDA(cudaMemset(v_prev, 0, size));
    CUDA(cudaMemset(w_prev, 0, size));
    CUDA(cudaMemset(dens, 0, size));
    CUDA(cudaMemset(dens_prev, 0, size));
}

// Free allocated memory
void free_data() {
    CUDA(cudaFree(u));
    CUDA(cudaFree(v));
    CUDA(cudaFree(w));
    CUDA(cudaFree(u_prev));
    CUDA(cudaFree(v_prev));
    CUDA(cudaFree(w_prev));
    CUDA(cudaFree(dens));
    CUDA(cudaFree(dens_prev));

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

    for (const auto &event: events) {
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
        CUDA(update_dens<<<1, 1>>>(dens, idx, density));
    }

    if (force_updated) {
        CUDA(update_uvw<<<1, 1>>>(u, v, w, idx, fx, fy, fz));
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
    CUDA(cudaMemcpy(dens_res, dens, size, cudaMemcpyDeviceToHost));
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
