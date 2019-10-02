#include "ComputeShader.h"

#include <Engine/Utility/FileReader.h>

LaiEngine::ComputeShader::ComputeShader()
{
	auto computeShaderStr = Utility::FileReader("Assets/Shaders/compute.cs");
	const GLchar * str = computeShaderStr.c_str();

	auto shaderID = IShader::Compile(computeShaderStr.c_str(), GL_COMPUTE_SHADER);
	m_programId = glCreateProgram();

	glAttachShader(m_programId, shaderID);
	glLinkProgram(m_programId);

	glDeleteShader(shaderID);

	GetUniforms();
}

void LaiEngine::ComputeShader::SetRandomSeed(const int seed)
{
	SetUniformValue(m_locationRandomSeed, seed);
}

void LaiEngine::ComputeShader::SetNumFrames(const int num)
{
	SetUniformValue(m_locationNumFrames, num);
}

void LaiEngine::ComputeShader::SetCameraPosition(const glm::vec3 & pos)
{
	SetUniformValue(m_locationCameraPosition, pos);
}

void LaiEngine::ComputeShader::SetCameraLowerLeftCorner(const glm::vec3 & v)
{
	SetUniformValue(m_locationCameraLowerLeftCorner, v);
}

void LaiEngine::ComputeShader::SetCameraHorizontal(const glm::vec3 & v)
{
	SetUniformValue(m_locationCameraHorizontal, v);
}

void LaiEngine::ComputeShader::SetCameraVertical(const glm::vec3 & v)
{
	SetUniformValue(m_locationCameraVertical, v);
}

void LaiEngine::ComputeShader::GetUniforms()
{
	UseProgram();

	m_locationRandomSeed = glGetUniformLocation(m_programId, "randomSeed");
	m_locationNumFrames = glGetUniformLocation(m_programId, "numFrames");

	m_locationCameraPosition = glGetUniformLocation(m_programId, "camera.Position");
	m_locationCameraLowerLeftCorner = glGetUniformLocation(m_programId, "camera.LowerLeftCorner");
	m_locationCameraHorizontal = glGetUniformLocation(m_programId, "camera.Horizontal");
	m_locationCameraVertical = glGetUniformLocation(m_programId, "camera.Vertical");
}
