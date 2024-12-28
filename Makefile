CXX = nvcc -std=c++11
SRCS = main.cu fluid_solver.cu EventManager.cpp
CFLAGS = -O3

all:
	$(CXX) $(CFLAGS) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm -f fluid_sim* gmon.out fluid_solver.s main.gprof output.* prof_md
	@echo Done.

copy-to-search:
	scp -p runcuda.sh $(SRCS) EventManager.h events.txt fluid_solver.h Makefile search:~/3dfluid/
	ssh search 'cd 3dfluid && module load gcc/7.2.0 && module load cuda/11.3.1 && make'

bench:
	echo "Starting to benchmark..."
	srun --partition=cpar --constraint=k20 ./runcuda.sh

bench-search: copy-to-search
	ssh search 'cd 3dfluid && make bench'
