
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define POP_SIZE 			32 // the suggested value is 40
#define SOLUTION_SIZE 		2
#define MAX_ITERATIONS 		10
#define FUNCTION 			SPHERE
#define CUDA_MAX_DOUBLE 	8.98847e+307 
#define MIN_VALUE			-100.0
#define MAX_VALUE			100.0

#define solution(var)		solutions[(blockIdx.x*SOLUTION_SIZE)+var]
#define objective			objectives[blockIdx.x]
#define rand(min,max)		(max-min)*curand_uniform(&states[blockIdx.x])+min

#define LOG_LEVEL			INFO

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
			value += ( solution(i) * solution(i) );
		}
	#endif

	return value;
}

__global__ void pso (curandState_t* states, double *solutions, double *objectives) {

	int tid = blockIdx.x;

	double velocity[SOLUTION_SIZE];
	double local_best[SOLUTION_SIZE]; // personal best
	double local_best_objective;

	// initialization
	for (int var = 0; var < SOLUTION_SIZE; ++var)
	{
		// initialize the swarm
		// http://cs.umw.edu/~finlayson/class/fall14/cpsc425/notes/23-cuda-random.html
		// x_i = U (min_d, max_d)
		solution(var) = rand(MIN_VALUE, MAX_VALUE);
		// v_i = U (mind - x_{i,d} ,maxd - x_{i,d})
		velocity[var] = rand(MIN_VALUE - solution(var), MAX_VALUE - solution(var));
		// p_i = x_i
		local_best[var] = solution(var); // the local_best is initialized with the solution
	}

	// evaluate solution
	objective = objective_function (solutions);

	// __syncthreads();
	// update global best
	// @TODO
	// Probably we need perform a reduce
	// So, the solution and objective need to be global

	for (int it = 0; it < MAX_ITERATIONS; ++it)
	{
		// @TODO
	}
	
	#if LOG_LEVEL == INFO
		printf("%d: x = [%lf %lf] v = [%lf %lf] = %lf\n", tid, solution(0), solution(1), velocity[0], velocity[1], objective);
	#endif
}

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              	&states[blockIdx.x]);
}

int main(int argc, char const *argv[])
{
	/* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
	curandState_t* states;
  	/* allocate space on the GPU for the random states */
	HANDLE_ERROR( cudaMalloc((void**) &states, POP_SIZE * sizeof(curandState_t) ) );
  	/* invoke the GPU to initialize all of the random states */
	init<<<POP_SIZE, 1>>>(time(0), states);

	// solutions[POPULATION_SIZE][SOLUTION_SIZE]
	// [p0v0 p0v1 p1v0 p1v1 p2v0 p2v1]
	double *dev_solutions_matrix; 
	double *dev_solutions_objectives;

	HANDLE_ERROR( cudaMalloc((void**) &dev_solutions_matrix, SOLUTION_SIZE * POP_SIZE * sizeof(double) ) );
	HANDLE_ERROR( cudaMalloc((void**) &dev_solutions_objectives, POP_SIZE * sizeof(double) ) );

	pso<<<POP_SIZE,1>>>(states, dev_solutions_matrix, dev_solutions_objectives);

	// cudaDeviceSynchronize is used to allow printf inside device functions
	// http://stackoverflow.com/questions/19193468/why-do-we-need-cudadevicesynchronize-in-kernels-with-device-printf
	cudaDeviceSynchronize();
	
	return 0;
}
