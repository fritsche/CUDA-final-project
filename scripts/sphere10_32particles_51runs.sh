#!/bin/bash

# USAGE: 
# * from CUDA-final-project directory type:
# ./scripts/sphere10_32particles_51runs.sh

echo "compiling for CPU"
gcc src/cpu_spso2011_multi_runs.c -o bin/cpu_spso2011_multi_runs -lm
cpu=./bin/cpu_spso2011_multi_runs

echo "compiling for GPU"
nvcc -arch=sm_20 src/gpu_spso2011_multi_blocks.cu -o bin/gpu_spso2011_multi_blocks
gpu=./bin/gpu_spso2011_multi_blocks

echo "executing for CPU"
echo "" > results/sphere10_32particles_cpu_multi_runs.txt
echo "" > results/sphere10_32particles_cpu_time_multi_runs.txt
for i in `seq 1 30`; do
	echo "CPU: $i"
	START=$(date +%s.%N)
	$cpu >> results/sphere10_32particles_cpu_multi_runs.txt
	END=$(date +%s.%N)
	DIFF=$(echo "$END - $START" | bc)
	echo $DIFF >> results/sphere10_32particles_cpu_time_multi_runs.txt
done

echo "executing for GPU"
echo "" > results/sphere10_32particles_gpu_multi_runs.txt
echo "" > results/sphere10_32particles_gpu_time_multi_runs.txt
for i in `seq 1 30`; do
	echo "GPU: $i"
	START=$(date +%s.%N)
	$gpu >> results/sphere10_32particles_gpu_multi_runs.txt
	END=$(date +%s.%N)
	DIFF=$(echo "$END - $START" | bc)
	echo $DIFF >> results/sphere10_32particles_gpu_time_multi_runs.txt
done
