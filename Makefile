CXX = nvcc -std=c++11
CFLAGS = -O3
SRCS = main.cu fluid_solver.cu EventManager.cpp

all:
	$(CXX) $(CFLAGS) $(SRCS) -o fluid_sim
