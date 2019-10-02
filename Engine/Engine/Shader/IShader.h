#pragma once

#include "Engine/Utility/NonCopyable.h"

#include <glm/matrix.hpp>
#include <GL/glew.h>

#include <string>

namespace LaiEngine
{
	class IShader : NonCopyable
	{
	public:

		IShader();
		IShader(const std::string& vertexFile, const std::string& fragmentFile);
		virtual ~IShader();

		void UseProgram() const;

		void SetUniformValue(GLuint location, int value);
		void SetUniformValue(GLuint location, float value);

		void SetUniformValue(GLuint location, const glm::vec2& vec);
		void SetUniformValue(GLuint location, const glm::vec3& vec);
		void SetUniformValue(GLuint location, const glm::vec4& vec);
		void SetUniformValue(GLuint location, const glm::mat4& matrix);

		GLuint m_programId = 0;
	protected:
		virtual void GetUniforms() = 0;


		static GLuint Load(const std::string& vertexShader, const std::string& fragmentShader);
		static GLuint Compile(const GLchar* source, GLenum shaderType);
		static GLuint Link(GLuint vertexShaderID, GLuint fragmentShaderID);
	};
}