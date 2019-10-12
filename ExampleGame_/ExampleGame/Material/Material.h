#pragma once

#include <ExampleGame/Core/Ray.h>
#include <ExampleGame/Core/Util.h>
#include <ExampleGame/GameObject/GameObject.h>
#include <glm/vec3.hpp>

#include <CUDA/cuda_runtime.h>
#include <CUDA/curand.h>
#include <CUDA/curand_kernel.h>


namespace LaiEngine
{
	class Ray;

	namespace Util
	{
		struct ShadeRec;
	}


	class Material
	{
	public:
		__device__ virtual bool Scatter(const Ray& ray_in, const Util::ShadeRec& rec, glm::vec3& attenuation, Ray& scattered, curandState *randState) const = 0;
	};


	class Lambertian : public Material
	{
	public:

		__device__ Lambertian(const glm::vec3& albedo) : Albedo(albedo) {};
		__device__ bool Scatter(const Ray& ray_in, const Util::ShadeRec& rec, glm::vec3& attenuation, Ray& scattered, curandState *randState) const override
		{
			glm::vec3 target = rec.p + rec.normal + CUDA::Util::GetRandomVecInUnitSphere(randState);
			scattered = Ray(rec.p, target - rec.p);
			attenuation = Albedo;

			return true;
		}

		glm::vec3 Albedo;
	};


	class Metal : public Material
	{
	public:

		__device__ Metal(const glm::vec3& albedo, float fuzz) : Albedo(albedo)
		{ 
			Fuzz = (fuzz < 1.0f) ? fuzz : 1;
		};

		__device__ bool Scatter(const Ray& ray_in, const Util::ShadeRec& rec, glm::vec3& attenuation, Ray& scattered, curandState *randState) const override
		{
			glm::vec3 reflected = CUDA::Util::Reflect(glm::normalize(ray_in.Direction), rec.normal);
			scattered = Ray(rec.p, reflected + Fuzz * CUDA::Util::GetRandomVecInUnitSphere(randState));
			attenuation = Albedo;

			return glm::dot(scattered.Direction, rec.normal) > 0.0f;
		}

		glm::vec3 Albedo;
		float Fuzz;
	};
}