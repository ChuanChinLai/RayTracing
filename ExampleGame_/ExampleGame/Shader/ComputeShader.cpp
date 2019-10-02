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

void LaiEngine::ComputeShader::SetNumFrames(const int num)
{
	SetUniformValue(m_locationNumFrames, num);
}

void LaiEngine::ComputeShader::SetInverseViewMat(const glm::mat4 & matrix)
{
	SetUniformValue(m_locationInverseViewMat, matrix);
}

void LaiEngine::ComputeShader::SetInverseProjectedMat(const glm::mat4 & matrix)
{
	SetUniformValue(m_locationInverseProjectedMat, matrix);
}

void LaiEngine::ComputeShader::GetUniforms()
{
	UseProgram();

	m_locationInverseViewMat = glGetUniformLocation(m_programId, "inverseViewMat");
	m_locationInverseProjectedMat = glGetUniformLocation(m_programId, "inverseProjectedMat");

	m_locationNumFrames = glGetUniformLocation(m_programId, "numFrames");
}
