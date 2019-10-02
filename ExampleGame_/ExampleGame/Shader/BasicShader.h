#pragma once

#include <Engine/Shader/IShader.h>
#include <glm/mat4x4.hpp>

namespace LaiEngine
{
	class BasicShader : public IShader
	{
	public:

		BasicShader(const std::string& vertexFile = "Assets/Shaders/basic.vs", 
			const std::string& fragmentFile = "Assets/Shaders/basic.fs");

		void SetModelMat(const glm::mat4& matrix);
		void SetViewMat(const glm::mat4& matrix);
		void SetProjectedMat(const glm::mat4& matrix);
			
	private:
		virtual void GetUniforms() override;

		GLuint m_locationModelMat;
		GLuint m_locationViewMat;
		GLuint m_locationProjectedMat;
	};
}