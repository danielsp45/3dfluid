SEARCH ?= 0
RUI ?= 0
CHICO ?= 0

CXX = nvcc -std=c++11
SRCS = main.cu fluid_solver.cu EventManager.cpp
CFLAGS = -O3 -Wno-deprecated-gpu-targets

ifeq ($(RUI),1)
    CFLAGS += -arch=sm_61
else ifeq ($(CHICO),1)
    CFLAGS += -arch=sm_75
else
	CFLAGS += -arch=sm_35
endif

# Default block dimensions
LIN_SOLVE_BLOCK_X ?= 32
LIN_SOLVE_BLOCK_Y ?= 4
LIN_SOLVE_BLOCK_Z ?= 1
THREADS_PER_BLOCK ?= 128

# Append block dimension macros to CFLAGS
CFLAGS += -D LIN_SOLVE_BLOCK_X=$(LIN_SOLVE_BLOCK_X) \
          -D LIN_SOLVE_BLOCK_Y=$(LIN_SOLVE_BLOCK_Y) \
          -D LIN_SOLVE_BLOCK_Z=$(LIN_SOLVE_BLOCK_Z) \
          -D THREADS_PER_BLOCK=$(THREADS_PER_BLOCK)

.PHONY: all clean run

all:
	$(CXX) $(CFLAGS) $(SRCS) -o fluid_sim

run:
	sbatch --partition day --constraint=k20 --ntasks=1 --time=2:00 --exclusive ./runcuda.sh

rrun:
	srun --partition day --constraint=k20 --ntasks=1 --time=2:00 --exclusive ./runcuda.sh

clean:
	@rm -f fluid_sim* gmon.out fluid_solver.s main.gprof output.* prof_md

copy-to-search:
	scp -p runcuda.sh test_thread_count_cuda.sh $(SRCS) EventManager.h events.txt fluid_solver.h cuda_utils.h Makefile search:~/3dfluid/

bench-search: copy-to-search
	ssh search "cd 3dfluid && module load gcc/7.2.0 && module load cuda/11.3.1 && make && make rrun"