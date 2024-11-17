CPP = g++ -Wall -std=c++11
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
CFLAGS = -O3 -funroll-loops -msse4 -mavx -ffast-math

THREADS ?= 24
MAX_THREADS ?= 48

all: seq par

seq:
	$(CPP) $(CFLAGS) -Wno-unknown-pragmas $(SRCS) -lm -o fluid_sim_seq

par:
	$(CPP) $(CFLAGS) -fopenmp $(SRCS) -lm -o fluid_sim

runseq: seq
	OMP_NUM_THREADS=1 ./fluid_sim_seq

runpar: par
	OMP_NUM_THREADS=$(THREADS) ./fluid_sim

clean:
	@echo Cleaning up...
	@rm -f fluid_sim* gmon.out fluid_solver.s main.gprof output.* prof_md
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
	scp -p *.sh $(SRCS) EventManager.h events.txt fluid_solver.h Makefile search:~/3dfluid/
	ssh search 'cd 3dfluid && module load gcc/11.2.0 && make'

bench:
	echo "Starting to benchmark..."
	srun --partition=day --constraint=c24 --ntasks=1 --cpus-per-task=$(MAX_THREADS) run.sh

bench-search: copy-to-search
	ssh search 'cd 3dfluid && make bench'
