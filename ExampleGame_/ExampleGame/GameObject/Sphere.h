#pragma once

#include "GameObject.h"

namespace LaiEngine
{
	class Sphere : public GameObject
	{
	public:
		__device__ Sphere() {};
		__device__ Sphere(const glm::vec3& pos, float r, Material* m) : Position(pos), Radius(r), pMaterial(m) {};
		__device__ bool Hit(const Ray& ray, float t_min, float t_max, Util::ShadeRec& rec) const override
		{
			glm::vec3 r = ray.Origin - Position;

			float a = glm::dot(ray.Direction, ray.Direction);
			float b = 2.0f * glm::dot(r, ray.Direction);
			float c = glm::dot(r, r) - Radius * Radius;

			float discriminant = b * b - 4.0f * a * c;

			//if (discriminant < 0.0f)
			//{
			//	rec.t = -1.0f;
			//	return false;
			//}
			//else 
			//{
			//	rec.t = (-b - std::sqrt(discriminant)) / (2.0f * a);
			//	return true;
			//}


			if (discriminant > 0.0f)
			{
				float temp = (-b - sqrt(discriminant)) / (2.0f * a);

				if (temp < t_max && temp > t_min)
				{
					rec.t = temp;
					rec.p = ray.GetPointAt(rec.t);
					rec.normal = (rec.p - Position) / Radius;
					rec.pMaterial = pMaterial;
					return true;
				}

				temp = (-b + sqrt(discriminant)) / (2.0f * a);

				if (temp < t_max && temp > t_min)
				{
					rec.t = temp;
					rec.p = ray.GetPointAt(rec.t);
					rec.normal = (rec.p - Position) / Radius;
					rec.pMaterial = pMaterial;
					return true;
				}
			}

			return false;
		}

		glm::vec3 Position;
		float Radius;
		Material *pMaterial;
	};
}