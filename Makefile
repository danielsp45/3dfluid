CPP = g++ -Wall -std=c++11 -O3
SRCS = main.cpp fluid_solver.cpp EventManager.cpp

all:
	$(CPP) $(SRCS) -o fluid_sim

run: all
	./fluid_sim

time: all
	time ./fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid
	@echo Done.
