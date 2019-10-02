#include "BasicShader.h"

LaiEngine::BasicShader::BasicShader(const std::string & vertexFile, const std::string & fragmentFile) : IShader(vertexFile, fragmentFile)
{
	GetUniforms();
}

void LaiEngine::BasicShader::SetModelMat(const glm::mat4 & matrix)
{
	SetUniformValue(m_locationModelMat, matrix);
}

void LaiEngine::BasicShader::SetViewMat(const glm::mat4 & matrix)
{
	SetUniformValue(m_locationViewMat, matrix);
}

void LaiEngine::BasicShader::SetProjectedMat(const glm::mat4 & matrix)
{
	SetUniformValue(m_locationProjectedMat, matrix);
}

void LaiEngine::BasicShader::GetUniforms()
{
	UseProgram();

	m_locationModelMat     = glGetUniformLocation(m_programId, "modelMat");
	m_locationViewMat      = glGetUniformLocation(m_programId, "viewMat");
	m_locationProjectedMat = glGetUniformLocation(m_programId, "projectedMat");
}
