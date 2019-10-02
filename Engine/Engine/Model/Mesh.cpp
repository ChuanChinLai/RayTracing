#include "Mesh.h"

LaiEngine::Cube::Cube()
{
	vertices.resize(8);

	// front
	vertices[0].Position.x = -0.5f;
	vertices[0].Position.y = -0.5f;
	vertices[0].Position.z =  0.5f;

	vertices[1].Position.x =  0.5f;
	vertices[1].Position.y = -0.5f;
	vertices[1].Position.z =  0.5f;

	vertices[2].Position.x =  0.5f;
	vertices[2].Position.y =  0.5f;
	vertices[2].Position.z =  0.5f;

	vertices[3].Position.x = -0.5f;
	vertices[3].Position.y =  0.5f;
	vertices[3].Position.z =  0.5f;

	// back
	vertices[4].Position.x = -0.5f;
	vertices[4].Position.y = -0.5f;
	vertices[4].Position.z = -0.5f;

	vertices[5].Position.x =  0.5f;
	vertices[5].Position.y = -0.5f;
	vertices[5].Position.z = -0.5f;

	vertices[6].Position.x =  0.5f;
	vertices[6].Position.y =  0.5f;
	vertices[6].Position.z = -0.5f;

	vertices[7].Position.x = -0.5f;
	vertices[7].Position.y =  0.5f;
	vertices[7].Position.z = -0.5f;


	indices = {
		// front
		0, 1, 2,
		2, 3, 0,
		// right
		1, 5, 6,
		6, 2, 1,
		// back
		7, 6, 5,
		5, 4, 7,
		// left
		4, 0, 3,
		3, 7, 4,
		// bottom
		4, 5, 1,
		1, 0, 4,
		// top
		3, 2, 6,
		6, 7, 3};
}

LaiEngine::Plane::Plane()
{
	vertices.resize(4);

	float size = 1.0f;

	// front
	vertices[0].Position.x = -size;
	vertices[0].Position.y = -size;
	vertices[0].u = 0.0f; 
	vertices[0].v = 0.0f;

	vertices[1].Position.x = size;
	vertices[1].Position.y = -size;
	vertices[1].u = 1.0f;
	vertices[1].v = 0.0f;


	vertices[2].Position.x = size;
	vertices[2].Position.y = size;
	vertices[2].u = 1.0f;
	vertices[2].v = 1.0f;


	vertices[3].Position.x = -size;
	vertices[3].Position.y = size;
	vertices[3].u = 0.0f;
	vertices[3].v = 1.0f;


	indices = {
		// front
		0, 1, 2,
		2, 3, 0,
	};
}
