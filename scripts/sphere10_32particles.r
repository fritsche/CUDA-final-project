
boxplot(sphere10_32particles_cpu$V11, sphere10_32particles_gpu$V11, sphere10_32particles_gpu_shared$V11, names=c("CPU", "GPU", "GPU __shared__"), main="SPHERE(10) - 32 particles", ylab="Fitness value")

boxplot(sphere10_32particles_cpu_time$V1, sphere10_32particles_gpu_time$V1, sphere10_32particles_gpu_shared_time$V1, names=c("CPU", "GPU", "GPU __shared__"), main="SPHERE(10) - 32 particles", ylab="Execution time (s)")
