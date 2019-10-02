#pragma once

#include <Core/Ray.h>
#include <Core/Utility.h>
#include <glm/vec3.hpp>

#include <vector>

namespace LaiEngine
{
	class Material;

	class GameObject 
	{
	public:

		virtual ~GameObject() {};

		virtual bool Hit(const Ray& ray, float t_min, float t_max, Util::ShadeRec& rec) const = 0;
	};


	class GameObjectList : public GameObject
	{
	public:

		GameObjectList(const std::vector<GameObject*>& objects) : Objects(objects) {};

		bool Hit(const Ray& ray, float t_min, float t_max, Util::ShadeRec& rec) const override;

	private:

		std::vector<GameObject*> Objects;
	};

}