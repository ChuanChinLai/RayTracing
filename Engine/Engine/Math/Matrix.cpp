#include "Matrix.h"

#include <glm/gtc/matrix_transform.hpp>

glm::mat4 LaiEngine::MakeModelMatrix(const glm::vec3 & position, const glm::vec3 & rotation)
{
	glm::mat4 matrix;

	matrix = glm::rotate(matrix, glm::radians(rotation.x), { 1, 0, 0 });
	matrix = glm::rotate(matrix, glm::radians(rotation.y), { 0, 1, 0 });
	matrix = glm::rotate(matrix, glm::radians(rotation.z), { 0, 0, 1 });

	matrix = glm::translate(matrix, position);

	return matrix;
}

glm::mat4 LaiEngine::MakeViewMatrix(const glm::vec3 & position, const glm::vec3 & rotation)
{
	glm::mat4 matrix(1.f);

	matrix = glm::rotate(matrix, glm::radians(rotation.x), { 1, 0, 0 });
	matrix = glm::rotate(matrix, glm::radians(rotation.y), { 0, 1, 0 });
	matrix = glm::rotate(matrix, glm::radians(rotation.z), { 0, 0, 1 });

	matrix = glm::translate(matrix, -position);

	return matrix;
}

glm::mat4 LaiEngine::MakeProjectionMatrix()
{
	float fov = 90.0f;
	return glm::perspective(glm::radians(fov), (float)800 / (float)800, 0.1f, 2000.0f);
}
