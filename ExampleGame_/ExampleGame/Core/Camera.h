#pragma once

#include "Ray.h"

#include <glm/vec3.hpp>

namespace LaiEngine
{
	class Camera
	{
	public:

		Camera();

		void Update(float dt);

		const glm::mat4& GetViewMatrix() const noexcept;
		const glm::mat4& GetProjectedMatrix() const noexcept;
		const glm::mat4& GetProjectedViewMatrix() const noexcept;


		Ray GetRay(float u, float v);

		glm::vec3 Velocity; 
		glm::vec3 Position;
		glm::vec3 Rotation;

	private:

		glm::mat4 m_ProjectedMatrix;
		glm::mat4 m_ViewMatrix;
		glm::mat4 m_ProjectedViewMatrx;
	};


	namespace CUDA
	{
		class Camera
		{
		public:
			__device__ Camera() 
			{

			};

			__device__ Ray GetRay(float u, float v)
			{
				return LaiEngine::Ray(Position, glm::vec3(-1, -1, -1) + u * glm::vec3(2, 0, 0) + v * glm::vec3(0, 2, 0) - Position);
			}

			glm::vec3 Position;
		};
	}
}