#include "CUDAExample.h"

#include <ExampleGame/CUDA/kernel.cuh>
#include <ExampleGame/Core/Camera.h>
#include <ExampleGame/Core/Util.h>
#include <ExampleGame/GameObject/GameObject.h>
#include <ExampleGame/GameObject/Sphere.h>
#include <ExampleGame/Material/Material.h>



#include <ctime>
#include <iostream>

#define checkCudaErrors(val) check_cuda1( (val), #val, __FILE__, __LINE__ )

void check_cuda1(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


void LaiEngine::CUDA::CUDAExample::Init(int nx, int ny, int tx, int ty)
{
	clock_t start, stop;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	start = clock();

	InitRandState(nx, ny, tx, ty);
	InitWorld();


	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";
}

void LaiEngine::CUDA::CUDAExample::Update(uint8_t* outputBuffer, curandState* randBuffer, int nx, int ny, int ns, int tx, int ty)
{
	clock_t start, stop;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	start = clock();

	// allocate device buffer
	uint8_t* deviceBuffer;
	constexpr size_t size_rgba = 4;
	const size_t buffer_size = nx * ny * size_rgba;
	checkCudaErrors(cudaMallocManaged((void **)&deviceBuffer, buffer_size));


	LaiEngine::CUDA::kernel::Render << < blocks, threads >> > (deviceBuffer, nx, ny, ns, camera, world, randState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(outputBuffer, deviceBuffer, buffer_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(deviceBuffer));

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";
}

void LaiEngine::CUDA::CUDAExample::Free()
{
	LaiEngine::CUDA::kernel::FreeWorld << <1, 1 >> > (gameobjects, world, camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(randState));

	checkCudaErrors(cudaFree(gameobjects));
	checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(camera));

	cudaDeviceReset();
}

void LaiEngine::CUDA::CUDAExample::InitRandState(const int nx, const int ny, const int tx, const int ty)
{
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	const size_t bufferSize = nx * ny * sizeof(curandState);

	checkCudaErrors(cudaMalloc((void **)&randState, bufferSize));
	LaiEngine::CUDA::kernel::InitRandState << <blocks, threads >> > (nx, ny, randState);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void LaiEngine::CUDA::CUDAExample::InitWorld()
{
	constexpr int numObjects = 4;
	checkCudaErrors(cudaMalloc((void **)&gameobjects, numObjects * sizeof(LaiEngine::GameObject*)));
	checkCudaErrors(cudaMalloc((void **)&world, sizeof(LaiEngine::GameObject*)));
	checkCudaErrors(cudaMalloc((void **)&camera, sizeof(LaiEngine::CUDA::Camera*)));

	LaiEngine::CUDA::kernel::CreateWorld << <1, 1>> > (gameobjects, world, camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
