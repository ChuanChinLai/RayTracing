#pragma once

#include "Ray.h"

#include <glm/vec3.hpp>

namespace LaiEngine
{
	class GameObject;
	class Material;

	namespace Util
	{
		struct ShadeRec
		{
			glm::vec3 p;
			glm::vec3 normal;

			float t;

			Material* pMaterial;
		};

		glm::vec3 GetColor(const Ray& ray, GameObject* world, int depth);

		glm::vec3 GetRandomVecInUnitSphere();

		float GetRandom();

		glm::vec3 Reflect(const glm::vec3& v, const glm::vec3& n);
	}
}