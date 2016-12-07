
boxplot(sphere10_32particles_cpu_multi_runs$V11, sphere10_32particles_gpu_multi_runs$V11, names=c("CPU", "GPU"), main="SPHERE(10) - 32 particles - 51 runs", ylab="Fitness value")

boxplot(sphere10_32particles_cpu_time_multi_runs$V1, sphere10_32particles_gpu_time_multi_runs$V1, names=c("CPU", "GPU"), main="SPHERE(10) - 32 particles - 51 runs", ylab="Execution time (s)")
