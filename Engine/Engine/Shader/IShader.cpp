#include "IShader.h"

#include <Engine/Utility/FileReader.h>

#include <GL/glew.h>
#include <External/glm/gtc/type_ptr.hpp>

LaiEngine::IShader::IShader()
{
}

LaiEngine::IShader::IShader(const std::string & vertexFile, const std::string & fragmentFile)
{
	m_programId = IShader::Load(vertexFile, fragmentFile);
	UseProgram();
}


LaiEngine::IShader::~IShader()
{
	glDeleteProgram(m_programId);
}


void LaiEngine::IShader::UseProgram() const
{
	glUseProgram(m_programId);
}


void LaiEngine::IShader::SetUniformValue(GLuint location, int value)
{
	glUniform1i(location, value);
}


void LaiEngine::IShader::SetUniformValue(GLuint location, float value)
{
	glUniform1f(location, value);
}


void LaiEngine::IShader::SetUniformValue(GLuint location, const glm::vec2 & vec)
{
	glUniform2f(location, vec.x, vec.y);
}


void LaiEngine::IShader::SetUniformValue(GLuint location, const glm::vec3 & vec)
{
	glUniform3f(location, vec.x, vec.y, vec.z);
}


void LaiEngine::IShader::SetUniformValue(GLuint location, const glm::vec4 & vec)
{
	glUniform4f(location, vec.x, vec.y, vec.z, vec.w);
}


void LaiEngine::IShader::SetUniformValue(GLuint location, const glm::mat4 & matrix)
{
	glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
}

void LaiEngine::IShader::GetUniforms()
{

}

GLuint LaiEngine::IShader::Load(const std::string & vertexShader, const std::string & fragmentShader)
{
	auto vertexSource = Utility::FileReader(vertexShader);
	auto fragmentSource = Utility::FileReader(fragmentShader);

	auto vertexShaderID = IShader::Compile(vertexSource.c_str(), GL_VERTEX_SHADER);
	auto fragmentShaderID = IShader::Compile(fragmentSource.c_str(), GL_FRAGMENT_SHADER);

	auto programID = IShader::Link(vertexShaderID, fragmentShaderID);

	glDeleteShader(vertexShaderID);
	glDeleteShader(fragmentShaderID);

	return programID;
}

GLuint LaiEngine::IShader::Compile(const GLchar * source, GLenum shaderType)
{
	auto shaderID = glCreateShader(shaderType);

	glShaderSource(shaderID, 1, &source, nullptr);
	glCompileShader(shaderID);

	GLint isSuccess = 0;
	GLchar infoLog[512];

	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &isSuccess);
	if (!isSuccess)
	{
		glGetShaderInfoLog(shaderID, 512, nullptr, infoLog);
		throw std::runtime_error("Unable to load a shader: " + std::string(infoLog));
	}

	return shaderID;
}

GLuint LaiEngine::IShader::Link(GLuint vertexShaderID, GLuint fragmentShaderID)
{
	auto programID = glCreateProgram();

	glAttachShader(programID, vertexShaderID);
	glAttachShader(programID, fragmentShaderID);

	glLinkProgram(programID);

	return programID;
}

