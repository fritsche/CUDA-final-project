Parallel SMPSO
==============
Gian M. Fritsche
----------------

This document presents the proposal for the final project of the CUDA course as the implementation of a parallel SMPSO.
The Speed-Constrained Multi-Objective Particle Swarm Optimization (SMPSO) is a metaheuristic for solving multi-objective problems.
The SMPSO is an adaptation of the mono-objective PSO for multi-objective problems.
In PSO each particle (solution) moves in the search space based on its experience (personal best) and the swarm experience (global best). The cooperative behavior of the particles guides the search towards the best solutions found so far and hopefully to an optimal solution of the problem.
Since all particles are independent it is possible to execute each particle in parallel.

To adapt a PSO for multi-objective problems (MOPs) some changes are required.
Fist, as in MOPs the solutions may be equivalents it is necessary a method selecting the global leader (global best) to be used to guide each particle; This method should provide convergence, but also divergence, to avoid that all solutions converge to the same point in the search space.
Also, it is necessary to store the non-dominated (equivalent) solutions somehow. The implementation can use a bounded repository,
when the repository is full, the archiving method selects the worse solution and removes it.
Alternatively, the algorithm can use an unbounded repository.

In this work, it is proposed to implement the mono-objective version first.
Perform some tests and evaluate performance.
There are two ideas for this implementation,
First: implement each step as a kernel, (the implementation can use Thrust library at some parts),
each kernel receives the population (and other required data) as input, perform the step, and return the population back.
At this stage, the idea is to implement all parts of the algorithm on GPU but without taking into account the performance.
Then, the algorithm may be optimized;
The second idea of implementation is use just one kernel.
Each particle is executed by one thread and performs each step of the algorithm independently.
For this kernel, some implementation details may affect the performance as the computational cost but also in the quality of the output. For example, the global best may be updated asynchronously (as soon as the particles are updated) or synchronously (when all particles are updated).

The next challenge is to implement a multi-objective version.
For that, it is necessary the allocation and maintenance of a repository for the non-dominated solutions.
There are two sources of difficulties, 
the first is the selection of the solutions inside the repository to be used as leaders,
moreover, the second is the selection of the solutions to be removed from the repository when it is full.
For the second issue, it is proposed to use an unbounded repository.

The `population_size` should be as big as the GPU capabilities can support without increasing the execution time.

Pseudo-code of SMPSO
-------------------

````
1:     initializeSwarm() 
2:     initializeLeadersArchive() 
3:     generation = 0 
4:     while generation < maxGenerations do 
5:         computeSpeed() // Eqs. 2 - 7
6:         updatePosition() // Eq. 1
7:         mutation() // Turbulence 
8:         evaluation()
9:         updateLeadersArchive()
10:        updateParticlesMemory()
11:        generation ++
12:    end while 
13:    returnLeadersArchive()

````

Data structures:
----------------

The **particles position** and the **particle velocity** can be matrices of double. Each row represents one particle, and each column represents one decision variable.
<!--The decision variables may have different lower and upper bounds 
(But for DTLZ and ZDT families the upper and lower bounds are the same for all problems lower=[0.0] and upper=[1.0].
So, if we only use those problems, we can generate all values at once.
In other problems, we can consider that the decision variables are normalized). -->

The algorithm initializes the particles position randomly (normalized between 0 and 1), and the velocity is a matrix of zeros.
The personal best is also a matrix, initialized with the particles position.
The global best (or repository) is an array (in mono-objective) or a matrix (in multi-objective).

In mono-objective, the quality of the solutions is a double value,
so, the quality of the population is an array of `population_size` positions.
Same way for the personal best. Also, the global best quality is a double value.

In multi-objective, the quality of each solution is an array of double.
So, the quality of the population is a matrix where each row is a solution, and each column is an objective value.
The same for the personal best, and the global best (repository).

1: Initialize the Swarm
-----------------------

Initialize the particles position (randomly); 
Moreover, initialize the velocity as a vector of zeros.
Then all the particles should be evaluated.
As the decision variables are not independent in this step, it is suggested to evaluate one particle (a matrix row) per thread.
In multi-objective, it is not clear if the objective functions can be assessed independently in parallel and this is something to check for each problem.

The personal best is equal to the particles position.
Each thread may initialize the position, velocity and personal best of one particle.
Alternatively, the algorithm may initialize the entire particles position matrix in parallel (one thread per cell, or using a library).
The algorithm may also initialize the velocity matrix in parallel.
Moreover, the personal best is a copy of the particles position so that the copy may be done in parallel as well.


2: Initialize Leaders Archive
-----------------------------

Initialize the leaders archive with the non-dominated solutions.
First, all solutions should be ranked (sorted by objective value);
Then, the `repository_size` best solutions compose the new repository.

3: Initialize generation counter
--------------------------------

An integer counter: `generation=0`.

4: Enter main loop
------------------

Then, it is executed the main loop of the algorithm for a maximum number of iterations.

5, 6: Compute speed and update position
------------------------------------

Those steps are sets of equations operated over the solutions.
As all particles and decision variables are independent, it is possible to perform the operations one thread per solution per dimension (the whole population matrix at once). 

7: Mutation
-----------

Only for the multi-objective version.
One thread per solution.

8: Evaluation
-------------

The fitness function(s) is an(a set of) equation(s) that has the solution as arguments.
Probably one thread per solution.
Perhaps one thread per solution per objective function (if possible).

9: Update leaders archive
-------------------------

The update leaders archive is the trickiest part of implementing in parallel for the multi-objective version.
First, all solutions are combined (repository + population);
Then, the algorithm removes the dominated solutions,
Finally, all remaining solutions should be ranked (sorted) accordingly to some criteria, and the `repository_size` best solutions compose the new repository.  The other solutions do not engage to the repository.

10: Update particles memory
---------------------------------------

Each particle (solution) compares its current position to its personal best information by dominance.
Then, if the new position is best or equivalent the personal best information is updated.
(This is one approach, it is possible to use other approaches instead.)
One thread per particle.

11: Update generation counter
-----------------------------

`generation++`

In the synchronized version, the counter counts generations. Alternatively, it counts fitness evaluations for the asynchronous version.

12, 13: Return leaders archive
------------------------------

Finally, the algorithm outputs the repository matrix (decision variables [VAR file] and the objective values [FUN file]).

#Conclusion

The proposal presented in this document is the implementation of a parallel Particle Swarm Optimization.
The minimal requirement for this project is the implementation of a functional parallel mono-objective version.
The multi-objective version is proposed as an extra stage to be implemented fully in parallel or, if the first is not possible, at least partially in parallel and partially sequential.

At each step, it will be compared the implemented version to the original sequential version to validation and performance assessment.
