#!/bin/bash

# USAGE: 
# * from CUDA-final-project directory type:
# ./scripts/sphere10_32particles_51runs_loop.sh

echo "compiling for CPU"
gcc src/cpu_spso2011.c -o bin/cpu_spso2011 -lm
cpuloop=./bin/cpu_spso2011

seeds=(274980940 760592088 947966310 467402166 268439617 535541803 162335949 621879666 766147508 647196725 793403018 737248219 260235147 364000053 795492773 503821177 557257207 684902972 268880461 111875254 550505820 179188275 805120258 182278575 735712554 675882616 702336295 147814176 174689199 158048576 763415466 547836017 742515573 778379193 941644558 959554064 337038737 442017622 266933211 739412884 519118939 124340923 198660993 650429366 878076118 94739280 622611244 55787391 368115853 558093724 455352700)

echo "executing for CPU single run executable"
echo "" > results/sphere10_32particles_cpu_time_multi_runs_loop.txt
for i in `seq 1 30`; do
	echo "CPU: $i"
	START=$(date +%s.%N)
	for seed in ${seeds[@]}; do
		echo "" > results/sphere10_32particles_cpu_multi_runs_loop_$seed-$i.txt
		$cpuloop $seed >> results/sphere10_32particles_cpu_multi_runs_loop_$seed-$i.txt &
	done

	# while there is cpu_spso2011 running
	aux=$(ps -eo cmd,etime | grep cpu_spso2011 | wc -l)
	while [[ "$aux" != "0" ]] ; do 
		aux=$(ps -eo cmd,etime | grep cpu_spso2011 | wc -l); 
	done

	END=$(date +%s.%N)
	DIFF=$(echo "$END - $START" | bc)
	echo $DIFF >> results/sphere10_32particles_cpu_time_multi_runs_loop.txt

done
