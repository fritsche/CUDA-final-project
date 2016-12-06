gpu_convergence = load("gpu_convergence.txt");
cpu_convergence = load("cpu_convergence.txt");

semilogx(mean(gpu_convergence), 'r-', 'linewidth', 1.5)
hold on
semilogx(mean(cpu_convergence), 'b-', 'linewidth', 1.5)
legend('GPU', 'CPU')
xlabel('iterations')
ylabel('fitness value')
xlabel('log(iterations)')
xlabel('iterations')
title('semilogx of mean CPU and GPU convergence')
print -depsc semilogx_convergence.eps

hold off
semilogy(mean(gpu_convergence), 'r-', 'linewidth', 1.5)
hold on
semilogy(mean(cpu_convergence), 'b-', 'linewidth', 1.5)
legend('GPU', 'CPU')
xlabel('iterations')
ylabel('fitness value')
xlabel('log(iterations)')
xlabel('iterations')
title('semilogy of mean CPU and GPU convergence')
print -depsc semilogy_convergence.eps

hold off
plot(mean(gpu_convergence), 'r-', 'linewidth', 1.5)
hold on
plot(mean(cpu_convergence), 'b-', 'linewidth', 1.5)
legend('GPU', 'CPU')
xlabel('iterations')
ylabel('fitness value')
xlabel('log(iterations)')
xlabel('iterations')
title('plot of mean CPU and GPU convergence')
print -depsc plot_convergence.eps

hold off
loglog(mean(gpu_convergence), 'r-', 'linewidth', 1.5)
hold on
loglog(mean(cpu_convergence), 'b-', 'linewidth', 1.5)
legend('GPU', 'CPU')
xlabel('iterations')
ylabel('fitness value')
xlabel('log(iterations)')
xlabel('iterations')
title('loglog of mean CPU and GPU convergence')
print -depsc loglog_convergence.eps
