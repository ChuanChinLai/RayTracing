#version 430

layout(location = 0) in vec3 i_position;
layout(location = 1) in vec2 i_uv;

uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projectedMat;

out vec2 vertexUV;

void main()
{
  gl_Position = vec4(i_position, 1.0);
  vertexUV = i_uv;
}