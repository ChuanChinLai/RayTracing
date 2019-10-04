#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>

#include <ExampleGame/Core/Utility.h>

#include <SFML/Graphics.hpp>

#include <fstream>
#include <iostream>
#include <stdio.h>


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) 
{
	if (result) 
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


__global__ void render(float *deviceBuffer, int max_x, int max_y)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x * 4 + i * 4;

	deviceBuffer[pixel_index + 0] = float(i) / max_x;
	deviceBuffer[pixel_index + 1] = float(j) / max_y;
	deviceBuffer[pixel_index + 2] = 0.2f;
}

// Helper function for using CUDA to add vectors in parallel.


void GetPixelColorWithCuda(float *buffer, const size_t buffer_size, int nx, int ny, int tx, int ty)
{
	float* deviceBuffer; 

	std::cerr << "Rendering a " << nx << "x" << ny << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	// allocate FB
	checkCudaErrors(cudaMallocManaged((void **)&deviceBuffer, nx * ny * 4));

	clock_t start, stop;
	start = clock();
	// Render our buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render << <blocks, threads >> > (deviceBuffer, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	checkCudaErrors(cudaMemcpy(buffer, deviceBuffer, buffer_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(deviceBuffer));
}