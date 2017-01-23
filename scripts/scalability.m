scalability_schafferf7_cpu_multi_runs = load("scalability_schafferf7_cpu_multi_runs.txt");
scalability_schafferf7_gpu_multi_runs = load("scalability_schafferf7_gpu_multi_runs.txt");

scalability_schafferf7_cpu_time_multi_runs = load("scalability_schafferf7_cpu_time_multi_runs.txt");
scalability_schafferf7_gpu_time_multi_runs = load("scalability_schafferf7_gpu_time_multi_runs.txt");


plot(mean(scalability_schafferf7_cpu_multi_runs), 'bo-', 'linewidth', 1.5)
hold on
plot(mean(scalability_schafferf7_gpu_multi_runs), 'r*-', 'linewidth', 1.5)
legend('CPU', 'GPU')
xlabel('population size (threads per block)')
ylabel('fitness value')
set(gca,'Xtick',1:5,'XTickLabel',{'32', '64', '96', '128', '160'})
title('Scalability of fitness value')
print -depsc scalability_schafferf7.eps
hold off

plot(mean(scalability_schafferf7_cpu_time_multi_runs), 'bo-', 'linewidth', 1.5)
hold on
plot(mean(scalability_schafferf7_gpu_time_multi_runs), 'r*-', 'linewidth', 1.5)
legend('CPU', 'GPU')
xlabel('population size (threads per block)')
ylabel('execution time')
set(gca,'Xtick',1:5,'XTickLabel',{'32', '64', '96', '128', '160'})
title('Scalability of execution time')
print -depsc scalability_schafferf7_time.eps
hold off
