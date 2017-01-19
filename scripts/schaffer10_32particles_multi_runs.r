
boxplot(schafferf7_32particles_cpu_multi_runs$V11, schafferf7_32particles_gpu_multi_runs$V11, names=c("CPU", "GPU"), main="SCHAFFER(10) - 32 particles - 51 runs", ylab="Fitness value")

boxplot(schafferf7_32particles_cpu_time_multi_runs$V1, schafferf7_32particles_gpu_time_multi_runs$V1, names=c("CPU", "GPU"), main="SCHAFFER(10) - 32 particles - 51 runs", ylab="Execution time (s)")
