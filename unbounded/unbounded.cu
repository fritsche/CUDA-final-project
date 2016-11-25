
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define POP_SIZE 100
#define SOLUTION_SIZE 2

// static void HandleError( cudaError_t err,
// 	const char *file,
// 	int line ) {
// 	if (err != cudaSuccess) {
// 		printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
// 			file, line );
// 		exit( EXIT_FAILURE );
// 	}
// }
// #define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void pso (curandState_t* states) {
	int tid = blockIdx.x;
	double solution[SOLUTION_SIZE];
	for (int i = 0; i < SOLUTION_SIZE; ++i)
	{
		solution[i] = curand_uniform(&states[blockIdx.x]); // rand (0.0 .. 1.0)
	}
	printf("%d %lf %lf\n", tid, solution[0], solution[1]);
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
	cudaMalloc((void**) &states, POP_SIZE * sizeof(curandState_t));
  	/* invoke the GPU to initialize all of the random states */
	init<<<POP_SIZE, 1>>>(time(0), states);

	pso<<<POP_SIZE,1>>>(states);
	cudaDeviceSynchronize();
	
	return 0;
}
