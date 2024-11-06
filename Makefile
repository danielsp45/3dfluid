CPP = g++ -Wall -std=c++11
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
CFLAGS = -O3 -funroll-loops -msse4 -mavx -ffast-math

THREADS ?= 4

all:
	$(CPP) $(CFLAGS) $(SRCS) -o fluid_sim -lm -fopenmp

runseq: all
	export OMP_NUM_THREADS=1
	./fluid_sim

runpar: all
	export OMP_NUM_THREADS=$(THREADS)
	./fluid_sim

clean:
	@echo Cleaning up...
	@rm -f fluid_sim gmon.out fluid_solver.s main.gprof output.* prof_md
	@echo Done.

PROF_FLAGS = -pg

prof:
	$(CPP) $(CFLAGS) $(PROF_FLAGS) $(SRCS) -o fluid_sim -lm -o prof_md

run-prof: prof
	./prof_md

graph-prof: run-prof
	gprof prof_md > main.gprof
	gprof2dot -o output.dot main.gprof
	rm gmon.out
	dot -Tpng -o output.png output.dot

copy-to-search:
	scp -p Makefile fluid_solver.cpp main.cpp EventManager.h EventManager.cpp fluid_solver.h events.txt search:~/3dfluid/
	ssh search 'cd 3dfluid && module load gcc/11.2.0 && make'

bench:
	echo "Starting to execute 3 times..."
	srun --partition=cpar --cpus-per-task=$(THREADS) perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim

bench-search: copy-to-search
	ssh search 'cd 3dfluid && make bench'
