#pragma once

#include <Engine/Shader/IShader.h>

namespace LaiEngine
{
	class ComputeShader : public IShader
	{
	public:

		ComputeShader();

		void SetNumFrames(const int num);
		void SetGetInputs(const int num);
		void SetMixingRatio(const float num);

		void SetInverseViewMat(const glm::mat4& matrix);
		void SetInverseProjectedMat(const glm::mat4& matrix);

	private:
		void GetUniforms() override;

		GLuint m_locationInverseViewMat;
		GLuint m_locationInverseProjectedMat;

		GLuint m_locationGetInputs;
		GLuint m_locationNumFrames;
		GLuint m_locationMixingRatio;
	};
}