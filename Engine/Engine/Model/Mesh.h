#pragma once

#include <glm/vec3.hpp>
#include <GL/glew.h>
#include <vector>

namespace LaiEngine
{
	struct sVertex
	{
		glm::vec3 Position;
		//glm::vec3 Color;
		float u; 
		float v;
	};

	class Mesh
	{
	public:

		virtual ~Mesh() {};

		std::vector<sVertex> vertices;
		std::vector<GLuint>  indices;
	};


	class Cube : public Mesh
	{
	public:
		Cube(); 
	};

	class Plane : public Mesh
	{
	public:
		Plane();
	};
}
