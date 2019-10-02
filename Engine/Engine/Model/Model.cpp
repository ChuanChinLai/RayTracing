#include "Model.h"

#include <glm/mat4x4.hpp>

#include <gl/GLU.h>
#include <cstdint>
#include <cassert>
#include <iostream>


LaiEngine::Model::~Model()
{
	Release();
}


void LaiEngine::Model::Init(const Mesh & mesh)
{
	CreateVAO();
	CreateVBO(mesh.vertices);
	CreateEBO(mesh.indices);
}

void LaiEngine::Model::BindVAO() const
{
	glBindVertexArray(mVertexArrayId);
}

void LaiEngine::Model::Release()
{
	ReleaseVAO();
	ReleaseVBO(); 
	ReleaseEBO();
}

const size_t& LaiEngine::Model::GetIndexCount() const
{
	return mIndexCount;
}

void LaiEngine::Model::CreateVAO()
{
	constexpr GLsizei arrayCount = 1; 
	glGenVertexArrays(arrayCount, &mVertexArrayId);

	try
	{
		const auto errorCode = glGetError();

		if (errorCode == GL_NO_ERROR)
		{
			glBindVertexArray(mVertexArrayId);
			const auto errorCode = glGetError();
			if (errorCode != GL_NO_ERROR)
			{
				throw std::runtime_error("OpenGL failed to bind a new vertex array");
			}
		}
		else
		{
			throw std::runtime_error("OpenGL failed to get an unused vertex array ID");
		}
	}
	catch (std::runtime_error& error)
	{
		std::cout << error.what() << std::endl; 
	}
}

void LaiEngine::Model::CreateVBO(const std::vector<sVertex>& vertices)
{
	try
	{
		//Create vertex buffer
		constexpr GLsizei bufferCount = 1;
		glGenBuffers(bufferCount, &mVertexBufferId);
		const auto errorCode = glGetError();

		if (errorCode == GL_NO_ERROR)
		{
			//Bind vertex buffer
			glBindBuffer(GL_ARRAY_BUFFER, mVertexBufferId);
			auto errorCode = glGetError();
			if (errorCode != GL_NO_ERROR)
			{
				throw std::runtime_error("OpenGL failed to bind a new vertex buffer");
			}

			//Allocate vertex buffer
			const auto bufferSize = vertices.size() * sizeof(sVertex);
			glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(bufferSize), vertices.data(), GL_STATIC_DRAW);
			errorCode = glGetError();
			if (errorCode != GL_NO_ERROR)
			{
				throw std::runtime_error("OpenGL failed to allocate the vertex buffer");
			}

			//Initialize vertex format

			const auto stride = static_cast<GLsizei>(sizeof(sVertex));

			//position x y z
			{
				constexpr GLuint vertexElementLocation = 0;
				constexpr GLint elementCount = 3;
				constexpr GLboolean notNormalized = GL_FALSE;
				GLvoid* offset = reinterpret_cast<GLvoid*>(0);
				glVertexAttribPointer(vertexElementLocation, elementCount, GL_FLOAT, notNormalized, stride, offset);
				const auto errorCode = glGetError();

				if (errorCode == GL_NO_ERROR)
				{
					glEnableVertexAttribArray(vertexElementLocation);
					const GLenum errorCode = glGetError();
					if (errorCode != GL_NO_ERROR)
					{
						std::string message = "OpenGL failed to enable the POSITION vertex attribute at location: " + vertexElementLocation;
						throw std::runtime_error(message.c_str());
					}
				}
				else
				{
					throw std::runtime_error("OpenGL failed to set the POSITION vertex attribute at location: " + vertexElementLocation);
				}
			}

			//r g b
			//{
			//	constexpr GLuint vertexElementLocation = 1;
			//	constexpr GLint elementCount = 3;
			//	constexpr GLboolean notNormalized = GL_FALSE;
			//	GLvoid* offset = reinterpret_cast<GLvoid*>(3 * sizeof(GLfloat));
			//	glVertexAttribPointer(vertexElementLocation, elementCount, GL_FLOAT, notNormalized, stride, offset);
			//	const auto errorCode = glGetError();

			//	if (errorCode == GL_NO_ERROR)
			//	{
			//		glEnableVertexAttribArray(vertexElementLocation);
			//		const GLenum errorCode = glGetError();
			//		if (errorCode != GL_NO_ERROR)
			//		{
			//			std::string message = "OpenGL failed to enable the RGB vertex attribute at location: " + vertexElementLocation;
			//			throw std::runtime_error(message.c_str());
			//		}
			//	}
			//	else
			//	{
			//		throw std::runtime_error("OpenGL failed to set the RGB vertex attribute at location: " + vertexElementLocation);
			//	}
			//}

			//uv
			{
				constexpr GLuint vertexElementLocation = 1;
				constexpr GLint elementCount = 2;
				constexpr GLboolean notNormalized = GL_FALSE;
				GLvoid* offset = reinterpret_cast<GLvoid*>(3 * sizeof(GLfloat));
				glVertexAttribPointer(vertexElementLocation, elementCount, GL_FLOAT, notNormalized, stride, offset);
				const auto errorCode = glGetError();

				if (errorCode == GL_NO_ERROR)
				{
					glEnableVertexAttribArray(vertexElementLocation);
					const GLenum errorCode = glGetError();
					if (errorCode != GL_NO_ERROR)
					{
						std::string message = "OpenGL failed to enable the RGB vertex attribute at location: " + vertexElementLocation;
						throw std::runtime_error(message.c_str());
					}
				}
				else
				{
					throw std::runtime_error("OpenGL failed to set the RGB vertex attribute at location: " + vertexElementLocation);
				}
			}
		}
		else
		{
			throw std::runtime_error("OpenGL failed to get an unused vertex buffer ID");
		}
	}
	catch (std::runtime_error& error)
	{
		std::cout << error.what() << std::endl;
	}
}

void LaiEngine::Model::CreateEBO(const std::vector<GLuint>& indices)
{
	mIndexCount = indices.size();

	try
	{
		constexpr GLsizei bufferCount = 1;
		glGenBuffers(bufferCount, &mIndexBufferId);
		auto errorCode = glGetError();
		if (errorCode == GL_NO_ERROR)
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBufferId);
			const auto errorCode = glGetError();
			if (errorCode != GL_NO_ERROR)
			{
				throw std::runtime_error("OpenGL failed to bind a new index buffer");
			}
		}
		else
		{
			throw std::runtime_error("OpenGL failed to get an unused index buffer ID");
		}


		const auto bufferSize = indices.size() * sizeof(GLuint);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(bufferSize), indices.data(), GL_STATIC_DRAW);

		errorCode = glGetError();
		if (errorCode != GL_NO_ERROR)
		{
			throw std::runtime_error("OpenGL failed to allocate the index buffer");
		}
	}
	catch (std::runtime_error& error)
	{
		std::cout << error.what() << std::endl;
	}
}

void LaiEngine::Model::ReleaseVAO()
{
	try
	{
		if (mVertexArrayId != 0)
		{
			// Unbind the vertex array
			glBindVertexArray(0);
			const auto errorCode = glGetError();
			if (errorCode != GL_NO_ERROR)
			{
				throw std::runtime_error("OpenGL failed to unbind all vertex arrays before cleaning up geometry");
			}
		}

		constexpr GLsizei arrayCount = 1;
		glDeleteVertexArrays(arrayCount, &mVertexArrayId);
		const auto errorCode = glGetError();
		if (errorCode != GL_NO_ERROR)
		{
			throw std::runtime_error("OpenGL failed to delete the vertex array");
		}

		mVertexArrayId = 0;
	}
	catch (std::runtime_error& error)
	{
		std::cout << error.what() << std::endl;
	}
}

void LaiEngine::Model::ReleaseVBO()
{
	try
	{
		if (mVertexBufferId != 0)
		{
			constexpr GLsizei bufferCount = 1;
			glDeleteBuffers(bufferCount, &mVertexBufferId);
			const auto errorCode = glGetError();
			if (errorCode != GL_NO_ERROR)
			{
				throw std::runtime_error("OpenGL failed to delete the vertex buffer");
			}
			mVertexBufferId = 0;
		}
	}
	catch (std::runtime_error& error)
	{
		std::cout << error.what() << std::endl;
	}
}

void LaiEngine::Model::ReleaseEBO()
{
	try
	{
		if (mIndexBufferId != 0)
		{
			constexpr GLsizei bufferCount = 1;
			glDeleteBuffers(bufferCount, &mIndexBufferId);
			const auto errorCode = glGetError();
			if (errorCode != GL_NO_ERROR)
			{
				throw std::runtime_error("OpenGL failed to delete the index buffer");
			}
			mIndexBufferId = 0;
		}
	}
	catch (std::runtime_error& error)
	{
		std::cout << error.what() << std::endl;
	}
}

