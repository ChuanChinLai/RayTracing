#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;

	c[i] = a[i] + b[i];

}

__global__ void vector_add(double *a, double *b, double *c)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			c[index] = a[index] * a[index] + b[index] * b[index];
		}
	}
}


// Helper function for using CUDA to add vectors in parallel.

void addWithCuda(double *a, double *b, double *c, unsigned int size)
{
	int N = 1024 * 1024;

	double *d_a, *d_b, *d_c;

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);


	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	vector_add << < (N + (1024 - 1)) / 1024, 1024 >> > (d_a, d_b, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);


	printf("c[%d] = %f\n", 0, c[0]);
	printf("c[%d] = %f\n", N - 1, c[N - 1]);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}