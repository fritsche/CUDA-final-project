---

author:

- 'Gian M. Fritsche'

bibliography:

- 'bibliography.bib'

title: Parallel Standard Particle Swarm Optimization

...

Introduction
============

The Particle Swarm Optimization @PSO95 is a meta-heuristic based on the
behavior of bird flocks. Every iteration, each particle moves in the
search space based on three components.

-   <span>*Social:*</span> This component contributes for the
    exploitation of the algorithm. It guides the search towards the best
    solutions found by the swarm.

-   <span>*Individual:*</span> The individual component guides the
    particle to the best region that the particle has found.

-   <span>*Inertia:*</span> The inertia component increase the
    exploration of the algorithm. It is responsible for keeping the
    particle moving towards a previous direction and avoid abrupt
    changes of direction.

The position of one particle is a set of variable values,
<span>*i.e.*</span> a solution for an optimization problem. Then, the
algorithms evaluate the solution, and a fitness value is associated.
Based on the fitness value the Social and Individual components are
updated. Moreover, these components are used to compute the next
position of the particle (another solution for the problem).

Since its first publication, the PSO had proposed several adaptations
and improvements. In 2006 it was created the first version of a Standard
Particle Swarm Optimization @SPSO. The SPSO was not proposed to be the
best PSO available, but establish a common benchmark as a baseline to
assess the PSO variants in the literature.

Standard Particle Swarm optimization
------------------------------------

Since its first publication (2006), there is three versions of SPSO:
2006, 2007 and 2011. In the latest version the description is given as
follow (and showed in Algorithm \[alg:spso2011\]): The first population
(set of solutions) is initialized randomly in the search space. Then,
the algorithms initialize the velocity also randomly. The suggested
population size is $40$. The neighborhood topology used to update the
social component is the Adaptive Random Topology (ART). In the ART each
particle informs its current fitness to $K$ neighbors. If the fitness is
better than the previous social best information of the neighbor the
social best (fitness and position) is updated. A direct graph represents
the neighborhood, where each particle informs its quality to (at most)
$K$ neighbors. The graph is generated randomly initially and every time
that the best global fitness is not improved.

In the previous versions of PSO, the velocity was updated dimension by
dimension. However, in SPSO2011 the velocity is updated in a geometrical
way, that does not depend on the system of coordinates.

\[alg:spso2011\]

Parallel Standard Particle Swarm Optimization
=============================================

In this work, we use the SPSO2011. The parallel implementation
associates each particle with one thread (Algorithm \[alg:pspso2011\]).
In this way, all `foreach particle  i \in swarm ` were removed and
executed in parallel based on the thread id inside the block
($threadIdx.x$ or $tid$). In the create adaptive random neighborhood
method each particle selects up to $K$ neighbors. Then the particle
updates its position and velocity; the position is evaluated using the
fitness function, and the personal best is updated. Then, to update the
neighbor best information, the particle search on the adjacency matrix
of the neighborhood for all its neighbors and compares to itself. The
best global fitness is computed using atomic operations. The
$atomicMin(a, b)$ was implemented based on the CUDA $atomicCAS$, which
realizes a compare and swap operation. Before entering the main loop,
the implementation applies a thread synchronization.

The first particle copies the best fitness for later comparison. Inside
the main loop, the particle updates its velocity, position, and fitness.
The personal best information is updated. Before set the neighbor best
information a synchronization is necessary because the threads must
compare its fitness to the updated value of its neighbors. Then, the
neighbor best information is updated. Also, the best fitness is updated
using the $atomicMin$ function. After the best fitness update it is
applied another `__syncthreads()`. To wait for all threads update the
best fitness before using its value. If the best fitness is not
improved, then the new neighborhood matrix is generated. After $T$
iterations the solution with fitness equals to the best fitness is the
output of the algorithm.

\[alg:pspso2011\]

In the first implemented version it was not used shared memory, but in a
second implementation it was used and the execution time was improved.
Another item important to highlight is the two `__syncthreads()` inside
the loop. The use of `__syncthreads()` generally slow down the execution
of a CUDA program. However, in this implementation we did not have this
problem due to the block size equals to warp size. Since the threads in
the same warp execute in true (hardware) parallelism, there is no
situation where threads need to sync with others on the same block.

The first experiments were executed using only one block. However, the
experiments run the SPSO several times; those executions can be run in
parallel, using different and independent blocks. In the experiments
using only one block the CPU implementation had a better execution time.
However, when using multi executions at once, the GPU implementation
achieved a better execution time. The next section details the
experiments and results.

Experiments and Results
=======================

The first experiment was used to validate the equivalence regarding the
quality of the GPU and CPU implementation. At first, the convergence was
not similar between both implementations due to bugs on the CUDA
implementation. After fixing the bugs, both implementations showed
similar convergence along the execution.

For this, and all other, experiment it was used $T=3125$ iterations,
$32$ particles, $D=10$ decision variables, $K=3$, and the Sphere
function for fitness evaluation. We executed $51$ independent runs.
Those parameters were based on @SPSOCEC. Moreover, the Sphere function
is a simple optimization function present on the COCO (Comparing
Continuous Optimisers) benchmark [^1].

In the Figures \[fig:loglog\_convergence\]
and \[fig:loglog\_convergence\] it is presented the average convergence
of the algorithms (GPU and CPU) during the iterations. Where it is
possible to observe similar behavior.

[loglog of mean CPU and GPU convergence](img/loglog_convergence.eps)

[semilogy of mean CPU and GPU convergence](img/semilogy_convergence.eps)

Then, we implemented a version using shared memory also. The best-known
fitness, the previous best, the population (or swarm), the fitness
values and the adjacency matrix use shared memory. We compare the three
versions (CPU, GPU, and GPU with shared memory). In terms of convergence
the results were similar (as they should
\[Figure \[fig:sphere10\_32particles\_fitness\]\]). The experiments were
repeated $51$ times with the GPU versions using just one block with 32
threads. The GPU with shared memory was faster than GPU without using
shared memory. But, the CPU implementation was the fastest
(Figure \[fig:sphere10\_32particles\_time\]).

[Convergence of different implementations (single block - 51
executions)](img/sphere10_32particles_fitness.eps)

[Execution time of different implementations (single block - 51
executions)](img/sphere10_32particles_time.eps)

To use the CUDA parallelism capabilities and exploit the characteristics
of SPSO the GPU implementation executes the $51$ runs in parallel using
51 blocks. To evaluated the execution time we repeat the experiment30
times. The fitness of the GPU (with shared memory) and CPU continued to
be similar (Figure \[fig:sphere10\_32particles\_multi\_runs\_fitness\])
but the execution time of the GPU was faster than CPU
(Figure \[fig:sphere10\_32particles\_multi\_runs\_time\]).

[Convergence of different implementations (GPU 51-blocks, 30
repetitions)](img/sphere10_32particles_multi_runs_fitness.eps)

[Execution time of different implementations (GPU 51-blocks, 30
repetitions)](img/sphere10_32particles_multi_runs_time.eps)

Conclusions
===========

In this project, we implement a Parallel Standard Particle Swarm
Optimization. First, we compare the implementations regarding
convergence quality, in which both GPU and CPU implementations were
similar. After different implementations and experiments, it was
possible to achieve a 14.49 speed up. The execution time of the CPU
version for 51 runs was 15.45444 seconds (in an average of 30
repetitions). Moreover, the execution time for the GPU version was
1.065946 seconds.

[^1]: http://coco.gforge.inria.fr/
