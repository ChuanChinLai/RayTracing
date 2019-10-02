#include "Camera.h"

#include <Engine/Math/Matrix.h>

LaiEngine::Camera::Camera()
{
	Position = glm::vec3(0, 0, 5);
	m_ProjectedMatrix = MakeProjectionMatrix();
}

void LaiEngine::Camera::Update()
{
	m_ViewMatrix = MakeViewMatrix(Position, Rotation);
}

const glm::mat4& LaiEngine::Camera::GetViewMatrix() const noexcept
{
	return m_ViewMatrix;
}

const glm::mat4& LaiEngine::Camera::GetProjectedMatrix() const noexcept
{
	return m_ProjectedMatrix;
}

const glm::mat4& LaiEngine::Camera::GetProjectedViewMatrix() const noexcept
{
	return m_ProjectedViewMatrx;
}


LaiEngine::Ray LaiEngine::Camera::GetRay(float u, float v)
{
	return Ray();
}
