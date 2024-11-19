#!/bin/bash
#
#SBATCH --exclusive     # exclusive node for the job
#SBATCH --time=02:00    # allocation for 2 minutes

if [ "$1" == "stat" ]; then
    echo "Running perf stat..."
    for threads in {1..40}
    do
        export OMP_NUM_THREADS=$threads
        echo "Running with OMP_NUM_THREADS=$threads"
        perf stat -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim
    done
elif [ "$1" == "report" ]; then
    echo "Running perf record and generating report..."
    perf record ./fluid_sim
    perf report -n --stdio > perfreport
    echo "Perf report saved to 'perfreport'"
else
    echo "Usage: $0 {stat|report}"
    exit 1
fi
