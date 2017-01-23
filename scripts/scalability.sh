#!/bin/bash

# percorrer os arquivos executando-os

for pop in `seq 32 32 160`; do

	cpu=./bin/cpu_$pop
	gpu=./bin/gpu_$pop

	echo "executing for CPU $pop"
	echo "" > results/$pop/schafferf7_cpu_multi_runs.txt
	echo "" > results/$pop/schafferf7_cpu_time_multi_runs.txt
	for i in `seq 1 30`; do
		echo "CPU: $i"
		START=$(date +%s.%N)
		$cpu >> results/$pop/schafferf7_cpu_multi_runs.txt
		END=$(date +%s.%N)
		DIFF=$(echo "$END - $START" | bc)
		echo $DIFF >> results/$pop/schafferf7_cpu_time_multi_runs.txt
	done

	echo "executing for GPU $pop"
	echo "" > results/$pop/schafferf7_gpu_multi_runs.txt
	echo "" > results/$pop/schafferf7_gpu_time_multi_runs.txt
	for i in `seq 1 30`; do
		echo "GPU: $i"
		START=$(date +%s.%N)
		$gpu >> results/$pop/schafferf7_gpu_multi_runs.txt
		END=$(date +%s.%N)
		DIFF=$(echo "$END - $START" | bc)
		echo $DIFF >> results/$pop/schafferf7_gpu_time_multi_runs.txt
	done

done