#pragma once

#include <Core/Ray.h>
#include <Core/Utility.h>

#include <CUDA/cuda_runtime.h>
#include <glm/vec3.hpp>

#include <vector>

namespace LaiEngine
{
	class Material;

	class GameObject 
	{
	public:

		__device__ virtual ~GameObject() {};
		__device__ virtual bool Hit(const Ray& ray, float t_min, float t_max, Util::ShadeRec& rec) const = 0;
	};


	class GameObjectList : public GameObject
	{
	public:

		__device__ GameObjectList() {};
		__device__ GameObjectList(GameObject** list, int n) : mList(list), size(n) {};
		__device__ bool Hit(const Ray& ray, float t_min, float t_max, Util::ShadeRec& rec) const override
		{
			Util::ShadeRec tempRec;
			bool hitAnything = false;

			float closest_so_far = t_max;

			for (int i = 0; i < size; i++)
			{
				if (mList[i] != nullptr && mList[i]->Hit(ray, t_min, closest_so_far, tempRec))
				{
					hitAnything = true;
					closest_so_far = tempRec.t;
					rec = tempRec;
				}
			}

			return hitAnything;
		}

	private:
		GameObject** mList;
		int size; 
	};

}