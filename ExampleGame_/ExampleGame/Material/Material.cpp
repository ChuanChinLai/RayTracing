#include "Material.h"

#include <Core/Utility.h>

bool LaiEngine::Lambertian::Scatter(const Ray & ray_in, const Util::ShadeRec & rec, glm::vec3 & attenuation, Ray & scattered) const
{
	glm::vec3 target = rec.p + rec.normal + Util::GetRandomVecInUnitSphere();
	scattered = Ray(rec.p, target - rec.p);
	attenuation = Albedo;
	
	return true;
}

bool LaiEngine::Metal::Scatter(const Ray & ray_in, const Util::ShadeRec & rec, glm::vec3 & attenuation, Ray & scattered) const
{
	glm::vec3 reflected = Util::Reflect(glm::normalize(ray_in.Direction), rec.normal);
	scattered = Ray(rec.p, reflected + Fuzz * Util::GetRandomVecInUnitSphere());
	attenuation = Albedo;

	return glm::dot(scattered.Direction, rec.normal) > 0.0f;
}
