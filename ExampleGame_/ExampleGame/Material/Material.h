#pragma once

#include <glm/vec3.hpp>

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
		virtual bool Scatter(const Ray& ray_in, const Util::ShadeRec& rec, glm::vec3& attenuation, Ray& scattered) const = 0;
	};


	class Lambertian : public Material
	{
	public:

		Lambertian(const glm::vec3& albedo) : Albedo(albedo) {};
		bool Scatter(const Ray& ray_in, const Util::ShadeRec& rec, glm::vec3& attenuation, Ray& scattered) const override;

		glm::vec3 Albedo;
	};


	class Metal : public Material
	{
	public:

		Metal(const glm::vec3& albedo, float fuzz) : Albedo(albedo) 
		{ 
			Fuzz = (fuzz < 1.0f) ? fuzz : 1;
		};
		bool Scatter(const Ray& ray_in, const Util::ShadeRec& rec, glm::vec3& attenuation, Ray& scattered) const override;

		glm::vec3 Albedo;
		float Fuzz;
	};
}