
#include <stdio.h>
#include <stdlib.h>  // rand(), srand()
#include <time.h>    // time()
#include <float.h>
#include <math.h>

#define POP_SIZE 			32 // the suggested value is 40
#define SOLUTION_SIZE 		10
#define MIN_VALUE			-100.0
#define MAX_VALUE			100.0
#define FUNCTION 			SCHAFFER
#define MAX_ITERATIONS 		3125
#define K 					3
#define INDEPENDENT_RUNS	51

double objective_function (double *solution) {

	double value;

	#if FUNCTION == SPHERE
	value = 0.0;
	for (int i = 0; i < SOLUTION_SIZE; ++i)
	{
		value += ( solution[i] * solution[i] );
	}
	#endif

	// http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/schafferf7.html
	#if FUNCTION == SCHAFFER
		size_t i = 0;

		value = 0.0;
		for (i = 0; i < SOLUTION_SIZE - 1; ++i) {
			const double tmp = solution[i] * solution[i] + solution[i+1] * solution[i+1];
			value += pow(tmp, 0.25) * (1.0 + pow(sin(50.0 * pow(tmp, 0.1)), 2.0));
		}
		value = pow(value / ((double) (long) SOLUTION_SIZE - 1.0), 2.0);

	#endif

	return value;
}

void create_adaptive_random_neighborhood (char neighborhood_adjacency_matrix[POP_SIZE][POP_SIZE]) {
	for (int s = 0; s < POP_SIZE; ++s)
	{
		for (int i = 0; i < POP_SIZE; ++i)
		{
			neighborhood_adjacency_matrix[s][i] = 0;
		}
	}
	for (int s = 0; s < POP_SIZE; ++s)
	{
		neighborhood_adjacency_matrix[s][s] = 1;
		for (int i = 0; i < K; ++i)
		{
			int neighbor = rand()%POP_SIZE;
			// printf("%d %d\n", tid, neighbor);
			neighborhood_adjacency_matrix[s][neighbor] = 1;
		}
	}
}

char solutions_are_different (double *solution_a, double *solution_b) {
	char different = 0; 
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

double randn (double mu, double sigma)
{
	double U1, U2, W, mult;
	static double X1, X2;
	static int call = 0;

	if (call == 1)
	{
		call = !call;
		return (mu + sigma * (double) X2);
	}

	do
	{
		U1 = -1 + ((double) rand () / RAND_MAX) * 2;
		U2 = -1 + ((double) rand () / RAND_MAX) * 2;
		W = pow (U1, 2) + pow (U2, 2);
	}
	while (W >= 1 || W == 0);

	mult = sqrt ((-2 * log (W)) / W);
	X1 = U1 * mult;
	X2 = U2 * mult;

	call = !call;

	return (mu + sigma * (double) X1);
}

void rand_sphere (double *x, int dimension) {
	double length = 0;
	for (int i = 0; i < dimension; i++) {
		x[i] = 0.0;
	}

	for (int i = 0; i < dimension; i++) {
		x[i] = randn(0.0, 1.0);
		length += length + x[i] * x[i];
	}

	length = sqrt(length);

	double r = ((double)rand()/(RAND_MAX));

	for (int i = 0; i < dimension; i++) {
		x[i] = r * x[i] / length;
	}
}

void update_velocity (double *solution, double *local_best, double *neighbor_best, double *velocity) {

	double gravity_center[SOLUTION_SIZE];
	double random_solution[SOLUTION_SIZE];
	double c = 1.1931; // (1.0 / 2.0 + log(2) )
	double w = 0.72135; // 1.0 / (2.0 * log(2))

	if (solutions_are_different(local_best, neighbor_best)) {
		for (int var = 0; var < SOLUTION_SIZE; ++var)
		{
			gravity_center[var] = solution[var] + c * (local_best[var] + neighbor_best[var] - 2 * solution[var]) / 3.0;
		}
	} else {
		for (int var = 0; var < SOLUTION_SIZE; ++var)
		{
			gravity_center[var] = solution[var] + c * (local_best[var] - solution[var]) / 2.0;
		}
	}

	// radius equals to distance between gravity center and the solution
	double distance = 0;
	for (int var = 0; var < SOLUTION_SIZE; ++var)
	{
		distance += ( gravity_center[var] - solution[var] ) * ( gravity_center[var] - solution[var] );
	}
	double radius = sqrt(distance);

	double random[SOLUTION_SIZE];
	rand_sphere(random, SOLUTION_SIZE);

	for (int var = 0; var < SOLUTION_SIZE; ++var) {
		random_solution[var] = gravity_center[var] + radius * random[var];
	}

	for (int var = 0; var < SOLUTION_SIZE; ++var) {
		velocity[var] = w * velocity[var] + random_solution[var] - solution[var];
	}
}

void update_position (double *solution, double *velocity) 
{
	double change_velocity = -0.5 ;
	for (int var = 0; var < SOLUTION_SIZE; ++var) 
	{
		solution[var] = solution[var] + velocity[var];

		if (solution[var] < MIN_VALUE) 
		{
			solution[var] = MIN_VALUE;
			velocity[var] = change_velocity * velocity[var];
		}
		if (solution[var]  > MAX_VALUE) 
		{
			solution[var] = MAX_VALUE;
			velocity[var] = change_velocity * velocity[var];
		}
	}
}

double pso (double best_solution[SOLUTION_SIZE]) {

	double solutions[POP_SIZE][SOLUTION_SIZE];
	double objectives[POP_SIZE];
	double velocity[POP_SIZE][SOLUTION_SIZE];
	double local_best[POP_SIZE][SOLUTION_SIZE]; 
	double local_best_objective[POP_SIZE];
	double neighbor_best[POP_SIZE][SOLUTION_SIZE]; 
	double neighbor_best_objective[POP_SIZE];
	double best_fitness;
	double previous_best_fitness;
	char neighborhood_adjacency_matrix[POP_SIZE][POP_SIZE];

	for (int i = 0; i < POP_SIZE; ++i)
	{
		neighbor_best_objective[i] = DBL_MAX;
	}
	best_fitness = DBL_MAX;

	// initialization
	for (int i = 0; i < POP_SIZE; ++i)
	{	
		for (int var = 0; var < SOLUTION_SIZE; ++var)
		{
			// initialize the swarm
			// x_i = U (min_d, max_d)
			solutions[i][var] = ((MAX_VALUE-MIN_VALUE)*((double)rand()/(RAND_MAX)))+MIN_VALUE;
			// v_i = U (mind - x_{i,d} ,maxd - x_{i,d})
			velocity[i][var] = (((MAX_VALUE - solutions[i][var])-(MIN_VALUE - solutions[i][var])) * 
				((double)rand()/(RAND_MAX)))+(MIN_VALUE - solutions[i][var]);
			// p_i = x_i
			local_best[i][var] = solutions[i][var]; // the local_best is initialized with the solution
		}

		// evaluate solution
		objectives[i] = objective_function (solutions[i]);
		// f(p_i) = f(x_i)
		local_best_objective[i] = objectives[i];

	}

	create_adaptive_random_neighborhood (neighborhood_adjacency_matrix);

	for (int s = 0; s < POP_SIZE; ++s)
	{
		// update neighbor best
		int best_index = -1;
		double best_value = neighbor_best_objective[s];
		for (int i = 0; i < POP_SIZE; ++i)
		{
			if ( neighborhood_adjacency_matrix[i][s] && objectives[i] <= best_value ) {
				best_index = i;
				best_value = objectives[i];
				// printf("%d %d %lf\n", tid, i, best_value);
			}
		}
		if (best_index != -1) {
			for (int i = 0; i < SOLUTION_SIZE; ++i)
			{
				neighbor_best[s][i] = solutions[best_index][i];
			}
		}
		neighbor_best_objective[s] = best_value;

		// update global best
		if (objectives[s] < best_fitness) {
			best_fitness = objectives[s];
		}
	}

	// it = 1 because the initialization also counts
	for (int it = 1; it < MAX_ITERATIONS; ++it)
	{
		for (int i = 0; i < POP_SIZE; ++i)
		{
			update_velocity(solutions[i], local_best[i], neighbor_best[i], velocity[i]);
			update_position (solutions[i], velocity[i]); 
			// evaluate solution
			objectives[i] = objective_function (solutions[i]);	
			// update local best
			if (objectives[i] < local_best_objective[i])
			{
				local_best_objective[i] = objectives[i];
				for (int var = 0; var < SOLUTION_SIZE; ++var)
				{
					local_best[i][var] = solutions[i][var];
				}
			}
		}

		previous_best_fitness = best_fitness;

		for (int s = 0; s < POP_SIZE; ++s)
		{
			// update neighbor best
			int best_index = -1;
			double best_value = neighbor_best_objective[s];
			for (int i = 0; i < POP_SIZE; ++i)
			{
				if ( neighborhood_adjacency_matrix[i][s] && objectives[i] <= best_value ) {
					best_index = i;
					best_value = objectives[i];
				}
			}
			if (best_index != -1) {
				for (int i = 0; i < SOLUTION_SIZE; ++i)
				{
					neighbor_best[s][i] = solutions[best_index][i];
				}
			}
			neighbor_best_objective[s] = best_value;

			// update global best
			if (objectives[s] < best_fitness) {
				best_fitness = objectives[s];
			}
		}

		// if there is no improvement of the best known fitness
		if ( ! (best_fitness < previous_best_fitness ) ) 
		{
			create_adaptive_random_neighborhood (neighborhood_adjacency_matrix);
		}

	}

	for (int i = 0; i < POP_SIZE; ++i)
	{
		if (local_best_objective[i] == best_fitness) {
			for (int var = 0; var < SOLUTION_SIZE; ++var)
			{
				best_solution[var] = local_best[i][var];
			}
			break;
		}
	}

	return best_fitness;

}

int main(int argc, char const *argv[])
{
	srand( time(0) );
	double best_solution[SOLUTION_SIZE];
	double best_fitness;

	for (int run = 0; run < INDEPENDENT_RUNS; ++run)
	{
		best_fitness = pso(best_solution);

		for (int i = 0; i < SOLUTION_SIZE; ++i)
		{
			printf("%g\t", best_solution[i]);
		}
		printf("%g\n", best_fitness);
	}

	return 0;
}
