#pragma once

#include "GameObject.h"

namespace LaiEngine
{
	class Sphere : public GameObject
	{
	public:
		Sphere() {};
		Sphere(const glm::vec3& pos, float r, Material* m) : Position(pos), Radius(r), pMaterial(m) {};

		bool Hit(const Ray& ray, float t_min, float t_max, Util::ShadeRec& rec) const override;

		glm::vec3 Position;
		float Radius;
		Material *pMaterial;
	};
}