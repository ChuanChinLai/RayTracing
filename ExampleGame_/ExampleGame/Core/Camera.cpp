#include "Camera.h"

#include <Engine/Math/Matrix.h>

LaiEngine::Camera::Camera() : LowerLeftCorner(-2.0f, -2.0f, -1.0f), Horizontal(4.0f, 0.0f, 0.0f), Vertical(0.0f, 4.0f, 0.0f)
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
	return Ray(Position, LowerLeftCorner + u * Horizontal + v * Vertical);
}
