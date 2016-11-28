
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define POP_SIZE 		32 // the suggested value is 40
#define SOLUTION_SIZE 	2
#define MAX_ITERATIONS 	10
#define FUNCTION 		SPHERE
#define CUDA_MAX_DOUBLE 8.98847e+307 

#define LOG_LEVEL		INFO

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

__device__ double objective_function (double * solution) {

	double value;

	#if FUNCTION == SPHERE
		value = 0.0;
		for (int i = 0; i < SOLUTION_SIZE; ++i)
		{
			value += ( solution[i] * solution[i] );
		}
	#endif

	return value;
}

__global__ void pso (curandState_t* states, double global_best [], double *global_best_objective) {

	int tid = blockIdx.x;
	
	double solution[SOLUTION_SIZE]; // particle position
	double objective;
	double velocity[SOLUTION_SIZE];
	double local_best[SOLUTION_SIZE];
	double local_best_objective;

	for (int i = 0; i < SOLUTION_SIZE; ++i)
	{
		solution[i] = curand_uniform(&states[blockIdx.x]); // rand (0.0 .. 1.0)
		velocity[i] = curand_uniform(&states[blockIdx.x]); // rand (0.0 .. 1.0)
		local_best[i] = solution[i]; // the local_best is initialized with the solution
	}

	#if LOG_LEVEL == INFO
		if ( blockIdx.x == 0 ) { 
			printf("%d: %lf\n", tid, *global_best_objective );
		}
	#endif

	// evaluate solution
	objective = objective_function (solution);

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
		printf("%d: [%lf %lf] = %lf\n", tid, solution[0], solution[1], objective);
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

	double *host_global_best_objective_initial_value =  (double*) malloc (sizeof(double));
	*host_global_best_objective_initial_value = CUDA_MAX_DOUBLE;

	double *dev_global_best;
	double *dev_global_best_objective;

	HANDLE_ERROR( cudaMalloc((void**) &dev_global_best, SOLUTION_SIZE * sizeof(double) ) );
	HANDLE_ERROR( cudaMalloc((void**) &dev_global_best_objective, sizeof(double) ) );
	HANDLE_ERROR( cudaMemcpy(dev_global_best_objective, host_global_best_objective_initial_value, sizeof(double), cudaMemcpyHostToDevice));


	pso<<<POP_SIZE,1>>>(states, dev_global_best, dev_global_best_objective);

	// cudaDeviceSynchronize is used to allow printf inside device functions
	// http://stackoverflow.com/questions/19193468/why-do-we-need-cudadevicesynchronize-in-kernels-with-device-printf
	cudaDeviceSynchronize();
	
	return 0;
}
