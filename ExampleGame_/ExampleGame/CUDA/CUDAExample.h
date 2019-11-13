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
			void Update(uint8_t* outputBuffer, int nx, int ny, int ns, int tx, int ty);
			void Free();

		private:

			void InitTextureBuffer(const int nx, const int ny);
			void InitRandState(const int nx, const int ny, const int tx, const int ty);
			void InitWorld();

			uint8_t* deviceTextureBuffer;
			curandState* randState;

			LaiEngine::GameObject ** gameobjects;
			LaiEngine::GameObject ** world;
			LaiEngine::CUDA::Camera ** camera;
		};

	}
}