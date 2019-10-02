#pragma once

#include <glm/mat4x4.hpp>

namespace LaiEngine
{
	glm::mat4 MakeModelMatrix(const glm::vec3& position, const glm::vec3& rotation);
	glm::mat4 MakeViewMatrix(const glm::vec3& position, const glm::vec3& rotation);
	glm::mat4 MakeProjectionMatrix();
}