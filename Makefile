SEARCH ?= 0
RUI ?= 0
CHICO ?= 0

CXX = nvcc -std=c++11
SRCS = main.cu fluid_solver.cu EventManager.cpp
CFLAGS = -O3

ifeq ($(SEARCH), 1)
	CFLAGS += -arch=sm_35
endif

ifeq ($(RUI), 1)
	CFLAGS += -arch=sm_61
endif

ifeq ($(CHICO), 1)
	CFLAGS += -arch=sm_75
endif

all:
	$(CXX) $(CFLAGS) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm -f fluid_sim* gmon.out fluid_solver.s main.gprof output.* prof_md
	@echo Done.

copy-to-search:
	scp -p runcuda.sh $(SRCS) EventManager.h events.txt fluid_solver.h cuda_utils.h Makefile search:~/3dfluid/
