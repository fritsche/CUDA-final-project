#!/bin/bash

pop=32

awk '{print $11}' results/$pop/schafferf7_gpu_multi_runs.txt > results/scalability_schafferf7_gpu_multi_runs.txt
cat results/$pop/schafferf7_gpu_time_multi_runs.txt > results/scalability_schafferf7_gpu_time_multi_runs.txt

awk '{print $11}' results/$pop/schafferf7_cpu_multi_runs.txt > results/scalability_schafferf7_cpu_multi_runs.txt
cat results/$pop/schafferf7_cpu_time_multi_runs.txt > results/scalability_schafferf7_cpu_time_multi_runs.txt

for pop in `seq 64 32 160`; do
	awk '{print $11}' results/$pop/schafferf7_gpu_multi_runs.txt > columns.aux
	cp results/scalability_schafferf7_gpu_multi_runs.txt results/intermediate.aux
	paste -d'\t' results/intermediate.aux columns.aux > results/scalability_schafferf7_gpu_multi_runs.txt
	cp results/scalability_schafferf7_gpu_time_multi_runs.txt results/intermediate.aux
	paste -d'\t' results/intermediate.aux results/$pop/schafferf7_gpu_time_multi_runs.txt > results/scalability_schafferf7_gpu_time_multi_runs.txt

	awk '{print $11}' results/$pop/schafferf7_cpu_multi_runs.txt > columns.aux
	cp results/scalability_schafferf7_cpu_multi_runs.txt results/intermediate.aux
	paste -d'\t' results/intermediate.aux columns.aux > results/scalability_schafferf7_cpu_multi_runs.txt
	cp results/scalability_schafferf7_cpu_time_multi_runs.txt results/intermediate.aux
	paste -d'\t' results/intermediate.aux results/$pop/schafferf7_cpu_time_multi_runs.txt > results/scalability_schafferf7_cpu_time_multi_runs.txt

done

