#include "Ray.h"

#include <GameObject/GameObject.h>

#include <glm/vec3.hpp>
#include <glm/gtx/normal.hpp>

glm::vec3 LaiEngine::Ray::GetPointAt(float t) const
{
	return Origin + t * Direction;
}
