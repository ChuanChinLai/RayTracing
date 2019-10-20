#pragma once

#include <glm/glm.hpp>

#include <CUDA/cuda_runtime.h>
#include <CUDA/curand.h>
#include <CUDA/curand_kernel.h>

#include <cstdint>

namespace LaiEngine
{
	class GameObject;

	namespace CUDA
	{
		class Camera;
	}

	namespace CUDA
	{
		class CUDAExample
		{
		public:
			void Init(int nx, int ny, int tx, int ty);
			void Update(uint8_t* outputBuffer, curandState* randBuffer, int nx, int ny, int ns, int tx, int ty);
			void Free();

		private:

			void InitRandState(const int nx, const int ny, const int tx, const int ty);
			void InitWorld();

			curandState* randState;

			LaiEngine::GameObject ** gameobjects;
			LaiEngine::GameObject ** world;
			LaiEngine::CUDA::Camera ** camera;
		};

	}
}