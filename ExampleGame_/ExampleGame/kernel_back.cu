#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <CUDA/curand.h>
#include <CUDA/curand_kernel.h>

#include <ExampleGame/Core/Utility.h>
#include <ExampleGame/Core/Camera.h>
#include <ExampleGame/GameObject/GameObject.h>
#include <ExampleGame/GameObject/Sphere.h>
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

__global__ void render_init(int max_x, int max_y, curandState *randState) 
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &randState[pixel_index]);
}


__device__ bool hit_sphere(const glm::vec3& center, float radius, const LaiEngine::Ray& r) 
{
	glm::vec3 oc = r.Origin - center;
	float a = glm::dot(r.Direction, r.Direction);
	float b = 2.0f * dot(oc, r.Direction);
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4.0f*a*c;
	return (discriminant > 0.0f);
}



__device__ glm::vec3 GetColor(const LaiEngine::Ray& r, LaiEngine::GameObject **world)
{
	LaiEngine::Util::ShadeRec rec; 

	if (world != nullptr && (*world)->Hit(r, 0.0f, 100000.0f, rec))
	{
		return 0.5f * glm::vec3(rec.normal.x + 1.0f, rec.normal.y + 1.0f, rec.normal.z + 1.0f);
	}


	glm::vec3 unit_direction = glm::normalize(r.Direction);
	float t = 0.5f * (unit_direction.y + 1.0f);
	return (1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0);
}

__global__ void render(uint8_t* outputBuffer, int max_x, int max_y, int ns, LaiEngine::CUDA::Camera** camera, LaiEngine::GameObject** world, curandState* randState)
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= max_x || j >= max_y) return;


	constexpr int sizePerPixel = 4;
	int bufferIndex = j * max_x * sizePerPixel + i * sizePerPixel;

	int randStateIndex = j * max_x + i;
	curandState local_rand_state = randState[randStateIndex];

	glm::vec3 color; 

	for (int s = 0; s < ns; s++)
	{
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(max_y - j + 1 + curand_uniform(&local_rand_state)) / float(max_y);

		LaiEngine::Ray ray = (*camera)->GetRay(u, v);

		color += GetColor(ray, world);
	}

	color /= static_cast<float>(ns);

	uint8_t ir = static_cast<uint8_t>(255.99f * color.x);
	uint8_t ig = static_cast<uint8_t>(255.99f * color.y);
	uint8_t ib = static_cast<uint8_t>(255.99f * color.z);

	outputBuffer[bufferIndex + 0] = ir;
	outputBuffer[bufferIndex + 1] = ig;
	outputBuffer[bufferIndex + 2] = ib;
	outputBuffer[bufferIndex + 3] = 255;
}

__global__ void create_world(LaiEngine::GameObject **d_list, LaiEngine::GameObject **d_world, LaiEngine::CUDA::Camera** d_camera)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) 
	{
		*(d_list) = new LaiEngine::Sphere(glm::vec3(0, 0, -1), 0.5, nullptr);
		*(d_list + 1) = new LaiEngine::Sphere(glm::vec3(0, -100.5, -1), 100, nullptr);

		*d_world = new LaiEngine::GameObjectList(d_list, 2);
		*d_camera = new LaiEngine::CUDA::Camera();
	}
}


__global__ void free_world(LaiEngine::GameObject **d_list, LaiEngine::GameObject **d_world, LaiEngine::CUDA::Camera** d_camera)
{
	delete *(d_list);
	delete *(d_list + 1);
	delete *d_world;
	delete *d_camera;
}


// Helper function for using CUDA to add vectors in parallel.

void InitCuda(int nx, int ny, int tx, int ty)
{
	clock_t start, stop;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	// allocate random state
	curandState *d_randState;
	checkCudaErrors(cudaMalloc((void **)&d_randState, nx * ny * sizeof(curandState)));

	start = clock();
	render_init << <blocks, threads >> > (nx, ny, d_randState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

}


void RenderWithCuda(uint8_t* outputBuffer, const size_t buffer_size, int nx, int ny, int tx, int ty)
{
	clock_t start, stop;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);


	// allocate random state
	curandState *d_randState;
	checkCudaErrors(cudaMalloc((void **)&d_randState, nx * ny * sizeof(curandState)));

	start = clock();
	render_init << <blocks, threads >> > (nx, ny, d_randState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";


	// allocate device buffer
	uint8_t* deviceBuffer;
	checkCudaErrors(cudaMallocManaged((void **)&deviceBuffer, buffer_size));

	// make our world of hitables
	LaiEngine::GameObject **d_List;
	checkCudaErrors(cudaMalloc((void **)&d_List, 2 * sizeof(LaiEngine::GameObject* )));
	LaiEngine::GameObject **d_World;
	checkCudaErrors(cudaMalloc((void **)&d_World, sizeof(LaiEngine::GameObject* )));
	LaiEngine::CUDA::Camera **d_Camera;
	checkCudaErrors(cudaMalloc((void **)&d_Camera, sizeof(LaiEngine::CUDA::Camera* )));


	create_world << <1, 1>> > (d_List, d_World, d_Camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	constexpr int ns = 10; 

	start = clock();
	render<<< blocks, threads >>>(deviceBuffer, nx, ny, ns, d_Camera, d_World, d_randState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";


	checkCudaErrors(cudaMemcpy(outputBuffer, deviceBuffer, buffer_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	free_world << <1, 1>> > (d_List, d_World, d_Camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(d_randState));
	checkCudaErrors(cudaFree(d_List));
	checkCudaErrors(cudaFree(d_World));
	checkCudaErrors(cudaFree(deviceBuffer));

	cudaDeviceReset();
}