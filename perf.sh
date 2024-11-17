#!/bin/bash

perf record  ./fluid_sim 
perf report -n --stdio > perfreport
