#pragma once

#include <glm/glm.hpp>

#include <CUDA/cuda_runtime.h>
#include <CUDA/curand.h>
#include <CUDA/curand_kernel.h>


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