
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define POP_SIZE 			32 // the suggested value is 40
#define SOLUTION_SIZE 		10
#define K 					3
#define MAX_ITERATIONS 		3125
#define INDEPENDENT_RUNS	51
#define FUNCTION 			SPHERE
#define CUDA_MAX_DOUBLE 	8.98847e+307 
#define MIN_VALUE			-100.0
#define MAX_VALUE			100.0

#define tid							threadIdx.x
#define bid							blockIdx.x
#define stateid						threadIdx.x+blockIdx.x*blockDim.x
#define solution(row,col)			solutions[(row*SOLUTION_SIZE)+col]
#define best_solution(row,col)		best_solution[(row*SOLUTION_SIZE)+col]
#define rand(min,max)				(max-min)*curand_uniform(&states[stateid])+min
#define int_rand(max)				curand(&states[stateid])%max;
#define adjacency_matrix(row,col)	neighborhood_adjacency_matrix[(row*POP_SIZE)+col]


static void HandleError( cudaError_t err,
	const char *file,
	int line ) {
	if (err != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
			file, line );
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ double objective_function (double *solutions) {

	double value;

	#if FUNCTION == SPHERE
	value = 0.0;
	for (int i = 0; i < SOLUTION_SIZE; ++i)
	{
		value += ( solution(tid,i) * solution(tid,i) );
	}
	#endif

	return value;
}

__device__ double update_neighbor_best (double neighbor_best_objective, bool * neighborhood_adjacency_matrix, double *objectives, double *neighbor_best, double *solutions) {
	int bestIndex = -1;
	double bestValue = neighbor_best_objective;
	for (int i = 0; i < POP_SIZE; ++i)
	{
		if ( adjacency_matrix(i, tid) && objectives[i] <= bestValue ) {
			bestIndex = i;
			bestValue = objectives[i];
		}
	}
	if (bestIndex != -1) {
		for (int i = 0; i < SOLUTION_SIZE; ++i)
		{
			neighbor_best[i] = solution(bestIndex,i);
		}
	}
	return bestValue;
}

__device__ void create_adaptive_random_neighborhood (curandState_t* states, bool * neighborhood_adjacency_matrix) {
	for (int i = 0; i < POP_SIZE; ++i)
	{
		adjacency_matrix(tid, i) = 0;
	}
	adjacency_matrix(tid, tid) = 1;
	for (int i = 0; i < K; ++i)
	{
		int neighbor = int_rand(POP_SIZE);
		adjacency_matrix(tid, neighbor) = 1;
	}

}

__device__ bool solutions_are_different (double *solution_a, double *solution_b) {
	bool different = 0; 
	for (int i = 0; i < SOLUTION_SIZE; ++i)
	{
		if (solution_a[i] != solution_b[i])
		{
			different = 1;
			break;
		}
	}
	return different;
}

__device__ void rand_sphere (curandState_t* states, double *x, int dimension) {
	double length = 0;
	for (int i = 0; i < dimension; i++) {
		x[i] = 0.0;
	}

	for (int i = 0; i < dimension; i++) {
		x[i] = curand_normal (&states[stateid]);
		length += length + x[i] * x[i];
	}

	length = sqrt(length);

	double r = curand_uniform (&states[stateid]);

	for (int i = 0; i < dimension; i++) 
		{		x[i] = r * x[i] / length;
		}
	}

	__device__ void update_velocity (curandState_t* states, double *solutions, double *local_best, double *neighbor_best, double *velocity) {

		double gravity_center[SOLUTION_SIZE];
		double random_solution[SOLUTION_SIZE];
	double c = 1.1931; // (1.0 / 2.0 + log(2) )
	double w = 0.72135; // 1.0 / (2.0 * log(2))

	if (solutions_are_different(local_best, neighbor_best)) {
		for (int var = 0; var < SOLUTION_SIZE; ++var)
		{
			gravity_center[var] = solution(tid,var) + c * (local_best[var] + neighbor_best[var] - 2 * solution(tid,var)) / 3.0;
		}
	} else {
		for (int var = 0; var < SOLUTION_SIZE; ++var)
		{
			gravity_center[var] = solution(tid,var) + c * (local_best[var] - solution(tid,var)) / 2.0;
		}
	}

	// radius equals to distance between gravity center and the solution
	double distance = 0;
	for (int var = 0; var < SOLUTION_SIZE; ++var)
	{
		distance += ( gravity_center[var] - solution(tid,var) ) * ( gravity_center[var] - solution(tid,var) );
	}
	double radius = sqrt(distance);

	double random[SOLUTION_SIZE];
	rand_sphere(states, random, SOLUTION_SIZE);

	for (int var = 0; var < SOLUTION_SIZE; ++var) {
		random_solution[var] = gravity_center[var] + radius * random[var];
	}

	for (int var = 0; var < SOLUTION_SIZE; ++var) {
		velocity[var] = w * velocity[var] + random_solution[var] - solution(tid,var);
	}
}

__device__ void update_position (double *solutions, double *velocity) 
{
	double change_velocity = -0.5 ;
	for (int var = 0; var < SOLUTION_SIZE; ++var) 
	{
		solution(tid, var) = solution(tid, var) + velocity[var];

		if (solution(tid, var) < MIN_VALUE) 
		{
			solution(tid, var) = MIN_VALUE;
			velocity[var] = change_velocity * velocity[var];
		}
		if (solution(tid, var) > MAX_VALUE) 
		{
			solution(tid, var) = MAX_VALUE;
			velocity[var] = change_velocity * velocity[var];
		}
	}
}

__device__ double atomicMin(double* address, double val)
{
	unsigned long long int* address_as_ull =
	(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, 
			(val < __longlong_as_double(assumed)) ? __double_as_longlong(val) : assumed );

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}

__global__ void pso (curandState_t* states, double *global_best_objective, double* best_solution) {

	double velocity[SOLUTION_SIZE];
	double local_best[SOLUTION_SIZE]; // personal best
	double local_best_objective;
	double neighbor_best[SOLUTION_SIZE];
	double neighbor_best_objective = CUDA_MAX_DOUBLE;
	__shared__ double best_fitness;
	__shared__ double previous_best_fitness;
	__shared__ double solutions[POP_SIZE*SOLUTION_SIZE];
	__shared__ double objectives[POP_SIZE];
	__shared__ bool neighborhood_adjacency_matrix[POP_SIZE*POP_SIZE];

	// the first thread initializes the best known fitness 
	if (tid == 0)
	{
		best_fitness = CUDA_MAX_DOUBLE;
	}

	// initialization
	for (int var = 0; var < SOLUTION_SIZE; ++var)
	{
		// initialize the swarm
		// http://cs.umw.edu/~finlayson/class/fall14/cpsc425/notes/23-cuda-random.html
		// x_i = U (min_d, max_d)
		solution(tid,var) = rand(MIN_VALUE, MAX_VALUE);
		// v_i = U (mind - x_{i,d} ,maxd - x_{i,d})
		velocity[var] = rand(MIN_VALUE - solution(tid,var), MAX_VALUE - solution(tid,var));
		// p_i = x_i
		local_best[var] = solution(tid,var); // the local_best is initialized with the solution
	}

	// evaluate solution
	objectives[tid] = objective_function (solutions);
	// f(p_i) = f(x_i)
	local_best_objective = objectives[tid];

	// each thread creates its random neighborhood of size K
	create_adaptive_random_neighborhood (states, neighborhood_adjacency_matrix);

	// all particles need the objective value updated 
	__syncthreads(); 
	// update neighbor best
	neighbor_best_objective = update_neighbor_best (neighbor_best_objective, neighborhood_adjacency_matrix, objectives, neighbor_best, solutions);
	// update best known fitness
	atomicMin(&best_fitness, objectives[tid]);
	// wait all particles updates its neighbor best
	// wait all particles updates the best known fitness
	__syncthreads();

	// it = 1 because the initialization also counts
	for (int it = 1; it < MAX_ITERATIONS; ++it)
	{
		update_velocity (states, solutions, local_best, neighbor_best, velocity);
		update_position (solutions, velocity);
		// evaluate solution
		objectives[tid] = objective_function (solutions);

		// update local best
		if (objectives[tid] < local_best_objective)
		{
			local_best_objective = objectives[tid];
			for (int var = 0; var < SOLUTION_SIZE; ++var)
			{
				local_best[var] = solution(tid,var);
			}
		}

		// keep the previous best fitness for comparison
		if (tid == 0)
		{
			previous_best_fitness = best_fitness;
		}

		// all particles need the objective value updated 
		__syncthreads(); 
		// update neighbor best
		neighbor_best_objective = update_neighbor_best (neighbor_best_objective, neighborhood_adjacency_matrix, objectives, neighbor_best, solutions);
		// update best known fitness
		atomicMin(&best_fitness, objectives[tid]);
		// wait all particles updates its neighbor best
		// wait all particles updates the best known fitness
		__syncthreads();

		// if there is no improvement of the best known fitness
		if ( ! (best_fitness < previous_best_fitness ) ) 
		{
			// each thread creates its random neighborhood of size K
			create_adaptive_random_neighborhood (states, neighborhood_adjacency_matrix);
		}

	}

	// if the particle found the best solution
	if (best_fitness == local_best_objective)
	{
		global_best_objective[bid] = local_best_objective;
		for (int i = 0; i < SOLUTION_SIZE; ++i)
		{
			best_solution(bid,i) = local_best[i];
		}
	}

}

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

  	/* we have to initialize the state */
  	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              stateid, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              	&states[stateid]);}

int main(int argc, char const *argv[])
{
	/* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
	curandState_t* states;
  	/* allocate space on the GPU for the random states */
	HANDLE_ERROR( cudaMalloc((void**) &states, INDEPENDENT_RUNS * POP_SIZE * sizeof(curandState_t) ) );
  	/* invoke the GPU to initialize all of the random states */
	init<<<INDEPENDENT_RUNS, POP_SIZE>>>(time(0), states);

	// solutions[POPULATION_SIZE][SOLUTION_SIZE]
	// [p0v0 p0v1 p1v0 p1v1 p2v0 p2v1]
	// double *dev_solutions_matrix; 
	// double *dev_solutions_objectives;
	// bool *dev_neighborhood_adjacency_matrix;

	double *dev_best_solution;
	double *dev_global_best_objective;

	double *host_best_solution = (double*) malloc (sizeof(double) * SOLUTION_SIZE * INDEPENDENT_RUNS);
	double *host_global_best_objective = (double*) malloc (sizeof(double) * INDEPENDENT_RUNS);

	// HANDLE_ERROR( cudaMalloc((void**) &dev_neighborhood_adjacency_matrix, POP_SIZE * POP_SIZE * sizeof(bool) ) );
	// HANDLE_ERROR( cudaMalloc((void**) &dev_solutions_matrix, SOLUTION_SIZE * POP_SIZE * sizeof(double) ) );
	// HANDLE_ERROR( cudaMalloc((void**) &dev_solutions_objectives, POP_SIZE * sizeof(double) ) );
	HANDLE_ERROR( cudaMalloc((void**) &dev_global_best_objective, sizeof(double) * INDEPENDENT_RUNS ) );
	HANDLE_ERROR( cudaMalloc((void**) &dev_best_solution, SOLUTION_SIZE * sizeof(double) * INDEPENDENT_RUNS) );

	// pso<<<1, POP_SIZE>>>(states, dev_solutions_matrix, dev_solutions_objectives, dev_neighborhood_adjacency_matrix, dev_global_best_objective, dev_best_solution);

	pso<<<INDEPENDENT_RUNS, POP_SIZE>>>(states, dev_global_best_objective, dev_best_solution);


	HANDLE_ERROR( cudaMemcpy(host_global_best_objective, dev_global_best_objective, sizeof(double) * INDEPENDENT_RUNS, cudaMemcpyDeviceToHost));
	HANDLE_ERROR( cudaMemcpy(host_best_solution, dev_best_solution, sizeof(double) * SOLUTION_SIZE * INDEPENDENT_RUNS, cudaMemcpyDeviceToHost));

	for (int run = 0; run < INDEPENDENT_RUNS; ++run)
	{
		// printf("%d ", run);	
		for (int i = 0; i < SOLUTION_SIZE; ++i)
		{
			printf("%g\t", host_best_solution[(run*SOLUTION_SIZE)+i]);
		}
		printf("%g\n", host_global_best_objective[run]);
	}

	// cudaDeviceSynchronize is used to allow printf inside device functions
	// http://stackoverflow.com/questions/19193468/why-do-we-need-cudadevicesynchronize-in-kernels-with-device-printf
	cudaDeviceSynchronize();
	
	return 0;
}
