#include "Utility.h"

#include <GameObject/GameObject.h>
#include <Material/Material.h>

#include <glm/glm.hpp>


glm::vec3 LaiEngine::Util::GetColor(const Ray & ray, GameObject * world, int depth)
{
	ShadeRec rec;

	if (world && world->Hit(ray, 0.001f, std::numeric_limits<float>::max(), rec))
	{
		Ray scattered; 
		glm::vec3 attenuation;

		if (depth < 50 && rec.pMaterial->Scatter(ray, rec, attenuation, scattered))
		{
			return attenuation * GetColor(scattered, world, depth + 1);
		}
		
		return glm::vec3();

		//return 0.5f * glm::vec3(rec.normal.x + 1.0f, rec.normal.y + 1.0f, rec.normal.z + 1.0f);
		//glm::vec3 target = rec.p + rec.normal + GetRandomVecInSphere();
		//return 0.5f * GetColor(Ray(rec.p, target - rec.p), world, depth);
	}
	else
	{
		glm::vec3 unit_direction = glm::normalize(ray.Direction);
		float t = 0.5f * (unit_direction.y + 1.0f);
		return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
	}
}

glm::vec3 LaiEngine::Util::GetRandomVecInUnitSphere()
{
	glm::vec3 res;

	do
	{
		res = 2.0f * glm::vec3(GetRandom(), GetRandom(), GetRandom()) - glm::vec3(1.0f, 1.0f, 1.0f);
	} while (glm::length(res) >= 1.0f);


	return res;
}

float LaiEngine::Util::GetRandom()
{
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX + 1);
}

glm::vec3 LaiEngine::Util::Reflect(const glm::vec3 & v, const glm::vec3 & n)
{
	return v - 2.0f * glm::dot(v, n) * n;
}
