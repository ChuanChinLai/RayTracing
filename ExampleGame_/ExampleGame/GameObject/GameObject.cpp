#include "GameObject.h"

#include <Core/Utility.h>

bool LaiEngine::GameObjectList::Hit(const Ray & ray, float t_min, float t_max, Util::ShadeRec & rec) const
{
	Util::ShadeRec tempRec;
	bool hitAnything = false;

	float closest_so_far = t_max;

	for (int i = 0; i < Objects.size(); i++)
	{
		if (Objects[i] != nullptr && Objects[i]->Hit(ray, t_min, closest_so_far, tempRec))
		{
			hitAnything = true;
			closest_so_far = tempRec.t;
			rec = tempRec;
		}
	}

	return hitAnything;
}
