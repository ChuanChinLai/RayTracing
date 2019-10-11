
#include <glm/glm.hpp>

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
		void Init(int nx, int ny, int tx, int ty);
		void Update(uint8_t* buffer, const size_t buffer_size, int nx, int ny, int tx, int ty);


		__global__ void create_world(LaiEngine::GameObject **d_list, LaiEngine::GameObject **d_world, LaiEngine::CUDA::Camera** d_camera);

		__global__ void free_world(LaiEngine::GameObject **d_list, LaiEngine::GameObject **d_world, LaiEngine::CUDA::Camera** d_camera);

		__global__ void init_randState(int max_x, int max_y, curandState *randState);

		__device__ glm::vec3 get_colors(const LaiEngine::Ray& r, LaiEngine::GameObject **world);

		__device__ bool hit_sphere(const glm::vec3& center, float radius, const LaiEngine::Ray& ray);

		__global__ void render(uint8_t* outputBuffer, int max_x, int max_y, int ns, LaiEngine::CUDA::Camera** camera, LaiEngine::GameObject** world, curandState* randState);
	}
}