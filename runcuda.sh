#!/bin/sh
#
#SBATCH --time=02:00    # allocation for 2 minutes

nvprof ./fluid_sim
