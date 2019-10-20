#pragma once

#include <CUDA/cuda_runtime.h>
#include <CUDA/curand.h>
#include <CUDA/curand_kernel.h>

#include <cstdint>

namespace LaiEngine
{
	class Ray;
	class GameObject;

	namespace CUDA
	{
		class Camera;
	}


	namespace CUDA
	{
		namespace kernel
		{
			__global__ void InitRandState(int nx, int ny, curandState *randState);

			__global__ void CreateWorld(LaiEngine::GameObject **objects, LaiEngine::GameObject **world, LaiEngine::CUDA::Camera** camera);

			__global__ void FreeWorld(LaiEngine::GameObject **objects, LaiEngine::GameObject **world, LaiEngine::CUDA::Camera** camera);

			__global__ void Render(uint8_t* outputBuffer, int max_x, int max_y, int ns, LaiEngine::CUDA::Camera** camera, LaiEngine::GameObject** world, curandState* randState);
		}

	}


	namespace CUDA
	{
		__device__ glm::vec3 GetColors(const LaiEngine::Ray& ray, LaiEngine::GameObject **world, curandState *randState);
	}

}