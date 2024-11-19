#!/bin/bash
#
#SBATCH --exclusive     # exclusive node for the job
#SBATCH --time=02:00    # allocation for 2 minutes

if [ "$1" == "stat" ]; then
    echo "Running perf stat..."
    export OMP_NUM_THREADS=21
    perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
elif [ "$1" == "report" ]; then
    echo "Running perf record and generating report..."
    perf record ./fluid_sim
    perf report -n --stdio > perfreport
    echo "Perf report saved to 'perfreport'"
else
    echo "Usage: $0 {stat|report}"
    exit 1
fi
