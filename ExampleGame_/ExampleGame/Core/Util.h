#pragma once

#include <glm/glm.hpp>

#include <CUDA/cuda_runtime.h>
#include <CUDA/curand.h>
#include <CUDA/curand_kernel.h>

#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

namespace LaiEngine
{
	namespace CUDA
	{
		namespace Util
		{

			__device__ inline glm::vec3 GetRandomVecInUnitSphere(curandState * randState)
			{
				glm::vec3 vec;
				do
				{
					vec = 2.0f * glm::vec3(curand_uniform(randState), curand_uniform(randState), curand_uniform(randState)) - glm::vec3(1.0f, 1.0f, 1.0f);
				} while (glm::length(vec) >= 1.0f);

				return vec;
			}

			__device__ inline glm::vec3 Reflect(const glm::vec3& v, const glm::vec3& n)
			{
				return v - 2.0f * dot(v, n) * n;
			}
		}
	}
}