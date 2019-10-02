#pragma once

#include <glm/glm.hpp>
#include <glm/vec3.hpp>

namespace LaiEngine
{
	class GameObject;

	class Ray
	{
	public:

		Ray() {};

		Ray(const Ray& ray) : Origin(ray.Origin), Direction(ray.Direction) {};

		Ray(const glm::vec3& origin, const glm::vec3& direction) : Origin(origin), Direction(glm::normalize(direction)) { };

		glm::vec3 GetPointAt(float t) const;

		glm::vec3 Origin;
		glm::vec3 Direction;
	};
}