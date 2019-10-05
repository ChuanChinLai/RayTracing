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


__global__ void render(float *deviceBuffer, uint8_t* outputBuffer, int max_x, int max_y)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x * 4 + i * 4;

	deviceBuffer[pixel_index + 0] = static_cast<float>(i) / max_x;
	deviceBuffer[pixel_index + 1] = static_cast<float>(j) / max_y;
	deviceBuffer[pixel_index + 2] = 0.2;

	float r = deviceBuffer[pixel_index + 0];
	float g = deviceBuffer[pixel_index + 1];
	float b = deviceBuffer[pixel_index + 2];

	uint8_t ir = static_cast<uint8_t>(255.99f * r);
	uint8_t ig = static_cast<uint8_t>(255.99f * g);
	uint8_t ib = static_cast<uint8_t>(255.99f * b);

	outputBuffer[pixel_index + 0] = ir;
	outputBuffer[pixel_index + 1] = ig;
	outputBuffer[pixel_index + 2] = ib;
	outputBuffer[pixel_index + 3] = 255;
}

// Helper function for using CUDA to add vectors in parallel.


void GetPixelColorWithCuda(uint8_t* outputBuffer, const size_t buffer_size, int nx, int ny, int tx, int ty)
{
	// allocate device buffer
	float *deviceBuffer;
	checkCudaErrors(cudaMallocManaged((void **)&deviceBuffer, buffer_size * sizeof(float)));

	uint8_t* oBuffer;
	checkCudaErrors(cudaMallocManaged((void **)&oBuffer, buffer_size));

	clock_t start, stop;
	start = clock();
	// Render our buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render << <blocks, threads >> > (deviceBuffer, oBuffer, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	checkCudaErrors(cudaMemcpy(outputBuffer, oBuffer, buffer_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(deviceBuffer));
}