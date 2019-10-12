#pragma once

#include <glm/glm.hpp>
#include <glm/vec3.hpp>

#include <CUDA/cuda_runtime.h>

namespace LaiEngine
{
	class Ray
	{
	public:

		__device__ Ray() {};

		__device__ Ray(const Ray& ray) : Origin(ray.Origin), Direction(ray.Direction) {};

		__device__ Ray(const glm::vec3& origin, const glm::vec3& direction) : Origin(origin), Direction(glm::normalize(direction)) { };

		__device__ inline glm::vec3 GetPointAt(float t) const { return Origin + t * Direction; } ;

		glm::vec3 Origin;
		glm::vec3 Direction;
	};
}