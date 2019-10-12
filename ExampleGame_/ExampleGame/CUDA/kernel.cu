#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <CUDA/curand.h>
#include <CUDA/curand_kernel.h>


#include <ExampleGame/Core/Camera.h>
#include <ExampleGame/Core/Util.h>
#include <ExampleGame/GameObject/GameObject.h>
#include <ExampleGame/GameObject/Sphere.h>
#include <ExampleGame/Material/Material.h>


#include <ExampleGame/CUDA/kernel.cuh>

#include <SFML/Graphics.hpp>

#include <algorithm>
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



void LaiEngine::CUDA::Init(curandState* outputBuffer, const size_t bufferSize, int nx, int ny, int tx, int ty)
{
	clock_t start, stop;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	curandState *d_randState;
	checkCudaErrors(cudaMalloc((void **)&d_randState, bufferSize));

	start = clock();
	InitRandState << <blocks, threads >> > (nx, ny, d_randState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	checkCudaErrors(cudaMemcpy(outputBuffer, d_randState, bufferSize, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(d_randState));


	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";
}

void LaiEngine::CUDA::Update(uint8_t* outputBuffer, curandState* randBuffer, int nx, int ny, int ns, int tx, int ty)
{
	clock_t start, stop;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	start = clock();

	// allocate random state
	curandState *d_randState;
	checkCudaErrors(cudaMalloc((void **)&d_randState, nx * ny * sizeof(curandState)));
	checkCudaErrors(cudaMemcpy(d_randState, randBuffer, nx * ny * sizeof(curandState), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	// allocate device buffer
	uint8_t* deviceBuffer;
	constexpr size_t size_rgba = 4;
	const size_t buffer_size = nx * ny * size_rgba;
	checkCudaErrors(cudaMallocManaged((void **)&deviceBuffer, buffer_size));

	// make our world of hitables
	LaiEngine::GameObject **d_List;
	constexpr int numObjects = 4;
	checkCudaErrors(cudaMalloc((void **)&d_List, numObjects * sizeof(LaiEngine::GameObject*)));
	LaiEngine::GameObject **d_World;
	checkCudaErrors(cudaMalloc((void **)&d_World, sizeof(LaiEngine::GameObject*)));
	LaiEngine::CUDA::Camera **d_Camera;
	checkCudaErrors(cudaMalloc((void **)&d_Camera, sizeof(LaiEngine::CUDA::Camera*)));


	CreateWorld << <1, 1 >> > (d_List, d_World, d_Camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	Render << < blocks, threads >> > (deviceBuffer, nx, ny, ns, d_Camera, d_World, d_randState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(outputBuffer, deviceBuffer, buffer_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	FreeWorld << <1, 1 >> > (d_List, d_World, d_Camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(d_randState));
	checkCudaErrors(cudaFree(d_List));
	checkCudaErrors(cudaFree(d_World));
	checkCudaErrors(cudaFree(deviceBuffer));

	cudaDeviceReset();

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";
}

__global__ void LaiEngine::CUDA::CreateWorld(LaiEngine::GameObject ** d_list, LaiEngine::GameObject ** d_world, LaiEngine::CUDA::Camera ** d_camera)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		d_list[0] = new LaiEngine::Sphere(glm::vec3(0, 0, -1), 0.5f, new Lambertian(glm::vec3(0.8f, 0.3f, 0.3f)));
		d_list[1] = new LaiEngine::Sphere(glm::vec3(0, -100.5, -1), 100, new Lambertian(glm::vec3(0.8f, 0.8f, 0.0f)));
		d_list[2] = new LaiEngine::Sphere(glm::vec3(1.0f, 0, -1.0f), 0.5f, new Metal(glm::vec3(0.8f, 0.6f, 0.2f), 0.6f));
		d_list[3] = new LaiEngine::Sphere(glm::vec3(-1.0f, 0, -1.0f), 0.5f, new Metal(glm::vec3(0.8f, 0.8f, 0.8f), 0.1f));

		*d_world = new LaiEngine::GameObjectList(d_list, 4);
		*d_camera = new LaiEngine::CUDA::Camera();
	}
}

__global__ void LaiEngine::CUDA::FreeWorld(LaiEngine::GameObject ** d_list, LaiEngine::GameObject ** d_world, LaiEngine::CUDA::Camera ** d_camera)
{
	constexpr int numObjects = 4;

	for (int i = 0; i < numObjects; i++)
	{
		delete ((LaiEngine::Sphere *)d_list[i])->pMaterial;
		delete d_list[i];
	}

	delete *d_world;
	delete *d_camera;
}

__global__ void LaiEngine::CUDA::InitRandState(int max_x, int max_y, curandState * randState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &randState[pixel_index]);
}



__device__ glm::vec3 LaiEngine::CUDA::GetColors(const LaiEngine::Ray & ray, LaiEngine::GameObject ** world, curandState *randState)
{
	LaiEngine::Ray currentRay = ray;

	glm::vec3 attenuation = glm::vec3(1.0f);

	for (int i = 0; i < 50; i++)
	{
		LaiEngine::Util::ShadeRec rec;

		if (world != nullptr && (*world)->Hit(currentRay, 0.001f, 1000000.0f, rec))
		{
			LaiEngine::Ray scattered;
			glm::vec3 tempAttenuation;

			if (rec.pMaterial->Scatter(currentRay, rec, tempAttenuation, scattered, randState))
			{
				attenuation *= tempAttenuation;
				currentRay = scattered;
			}
			else
			{
				return glm::vec3(0.0f);
			}
			//glm::vec3 target = rec.p + rec.normal + CUDA::Util::GetRandomVecInUnitSphere(randState);
			//return 0.5f * glm::vec3(rec.normal.x + 1.0f, rec.normal.y + 1.0f, rec.normal.z + 1.0f);
		}
		else
		{
			glm::vec3 unit_direction = glm::normalize(currentRay.Direction);

			float t = 0.5f * (unit_direction.y + 1.0f);
			glm::vec3 c = (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);

			return attenuation * c;
		}
	}

	return glm::vec3(0.0f);
}


__global__ void LaiEngine::CUDA::Render(uint8_t * outputBuffer, int max_x, int max_y, int ns, LaiEngine::CUDA::Camera ** camera, LaiEngine::GameObject ** world, curandState * randState)
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

		color += GetColors(ray, world, &local_rand_state);
	}

	color /= static_cast<float>(ns);

	color.x = sqrt(color.x);
	color.y = sqrt(color.y);
	color.z = sqrt(color.z);

	uint8_t r = static_cast<uint8_t>(255.99f * color.x);
	uint8_t g = static_cast<uint8_t>(255.99f * color.y);
	uint8_t b = static_cast<uint8_t>(255.99f * color.z);


	outputBuffer[bufferIndex + 0] = r;
	outputBuffer[bufferIndex + 1] = g;
	outputBuffer[bufferIndex + 2] = b;
	outputBuffer[bufferIndex + 3] = 255;
}

