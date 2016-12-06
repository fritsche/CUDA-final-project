#!/bin/bash

echo "compiling for CPU"
gcc -Wall cpu_spso2011.c -o cpu_v0.3 -lm
cpu=./cpu_v0.3
echo "compiling for GPU"
nvcc -arch=sm_20 gpu_spso2011.cu -o gpu_v0.3
gpu=./gpu_v0.3
echo "compiling for GPU __shared__"
nvcc -arch=sm_20 gpu_spso2011_shared.cu -o gpu_shared_v0.3
gpu_shared=./gpu_shared_v0.3

# https://www.random.org/integers/
seeds=(274980940 760592088 947966310 467402166 268439617 535541803 162335949 621879666 766147508 647196725 793403018 737248219 260235147 364000053 795492773 503821177 557257207 684902972 268880461 111875254 550505820 179188275 805120258 182278575 735712554 675882616 702336295 147814176 174689199 158048576 763415466 547836017 742515573 778379193 941644558 959554064 337038737 442017622 266933211 739412884 519118939 124340923 198660993 650429366 878076118 94739280 622611244 55787391 368115853 558093724 455352700)

echo "" > sphere10_32particles_cpu.txt
echo "" > sphere10_32particles_cpu_time.txt
i=1
for seed in ${seeds[@]}; do
	echo "CPU: $i"
	START=$(date +%s.%N)
	$cpu $seed >> sphere10_32particles_cpu.txt
	END=$(date +%s.%N)
	DIFF=$(echo "$END - $START" | bc)
	echo $DIFF >> sphere10_32particles_cpu_time.txt
	((i++))
done

echo "" > sphere10_32particles_gpu.txt
echo "" > sphere10_32particles_gpu_time.txt
i=1
for seed in ${seeds[@]}; do
	echo "GPU: $i"
	START=$(date +%s.%N)
	$gpu $seed >> sphere10_32particles_gpu.txt
	END=$(date +%s.%N)
	DIFF=$(echo "$END - $START" | bc)
	echo $DIFF >> sphere10_32particles_gpu_time.txt
	((i++))
done

echo "" > sphere10_32particles_gpu_shared.txt
echo "" > sphere10_32particles_gpu_shared_time.txt
i=1
for seed in ${seeds[@]}; do
	echo "GPU __shared__: $i"
	START=$(date +%s.%N)
	$gpu_shared $seed >> sphere10_32particles_gpu_shared.txt
	END=$(date +%s.%N)
	DIFF=$(echo "$END - $START" | bc)
	echo $DIFF >> sphere10_32particles_gpu_shared_time.txt
	((i++))
done

