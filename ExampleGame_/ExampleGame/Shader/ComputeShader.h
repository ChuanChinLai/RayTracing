#pragma once

#include <Engine/Shader/IShader.h>

namespace LaiEngine
{
	class ComputeShader : public IShader
	{
	public:

		ComputeShader();

		void SetRandomSeed(const int seed);
		void SetNumFrames(const int num);

		void SetCameraPosition(const glm::vec3& pos);
		void SetCameraLowerLeftCorner(const glm::vec3& v);
		void SetCameraHorizontal(const glm::vec3& v);
		void SetCameraVertical(const glm::vec3& v);


	private:
		void GetUniforms() override;

		GLuint m_locationRandomSeed;
		GLuint m_locationNumFrames;
 
		GLuint m_locationCameraPosition; 
		GLuint m_locationCameraLowerLeftCorner;
		GLuint m_locationCameraHorizontal;
		GLuint m_locationCameraVertical;
	};
}