#pragma once

#include <Engine/Model/Mesh.h>

#include <Engine/Utility/NonCopyable.h>
#include <Engine/Utility/NonMovable.h>

namespace LaiEngine
{
	class Model : public NonMovable, public NonCopyable
	{
	public:

		Model() = default;
		~Model();

		void Init(const Mesh& mesh);

		void BindVAO() const;
		const size_t& GetIndexCount() const;

	private:


		void Release();

		void CreateVAO();
		void CreateVBO(const std::vector<sVertex>& vertices);
		void CreateEBO(const std::vector<GLuint>& indices);

		void ReleaseVAO();
		void ReleaseVBO();
		void ReleaseEBO();

		// A vertex array encapsulates the vertex data as well as the vertex input layout
		GLuint mVertexArrayId = 0;

		// A vertex buffer holds the data for each vertex
		GLuint mVertexBufferId = 0;
		GLuint mIndexBufferId = 0;

		size_t mIndexCount = 0;
	};
}