
#include <glm/glm.hpp>

#include <CUDA/cuda_runtime.h>
#include <CUDA/curand.h>
#include <CUDA/curand_kernel.h>

#include <cstdint>

namespace LaiEngine
{
	class Ray;
	class GameObject;

	class CUDAExample
	{
	public:
		void Init();
		void Update();
		void Free();
	private:
	};


	namespace CUDA
	{
		class Camera;
	}


	namespace CUDA
	{
		void Init(curandState* outputBuffer, const size_t bufferSize, int nx, int ny, int tx, int ty);
		void Update(uint8_t* outputBuffer, curandState* randBuffer, int nx, int ny, int ns, int tx, int ty);


		__global__ void CreateWorld(LaiEngine::GameObject **d_list, LaiEngine::GameObject **d_world, LaiEngine::CUDA::Camera** d_camera);

		__global__ void FreeWorld(LaiEngine::GameObject **d_list, LaiEngine::GameObject **d_world, LaiEngine::CUDA::Camera** d_camera);

		__global__ void InitRandState(int max_x, int max_y, curandState *randState);

		__device__ glm::vec3 GetColors(const LaiEngine::Ray& ray, LaiEngine::GameObject **world, curandState *randState);

		__global__ void Render(uint8_t* outputBuffer, int max_x, int max_y, int ns, LaiEngine::CUDA::Camera** camera, LaiEngine::GameObject** world, curandState* randState);
	}

}