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
	scp -p runcuda.sh $(SRCS) EventManager.h events.txt fluid_solver.h cuda_utils.h Makefile search:~/3dfluid/
