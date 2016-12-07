
#include <stdio.h>

__global__ void teste (int *dev_a) 
{
	__shared__ int a[10];
	
	a[threadIdx.x] = threadIdx.x;
	__syncthreads();
	printf("[%d] %d\n", threadIdx.x, a[(threadIdx.x+1)%10] );
}

int main(int argc, char const *argv[])
{
	int *dev_a;
	// int *host_a;
	cudaMalloc((void**) &dev_a, 10 * sizeof(int));
	teste <<<1,10>>> (dev_a);
	// host_a = (int*) malloc (2 * sizeof(int));
	// cudaMemcpy( host_a, dev_a, 2 * sizeof(int), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < 2; ++i)
	// {
	// 	printf("%d %d\n", i, host_a[i] );
	// }
	cudaDeviceSynchronize();
	return 0;
}
