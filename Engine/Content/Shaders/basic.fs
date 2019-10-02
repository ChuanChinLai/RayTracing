#version 430

in vec2 vertexUV;
uniform sampler2D img;

out vec4 o_color;

void main()
{	
	vec3 color = texture2D(img, vertexUV).rgb;
	o_color = vec4(color, 1.0f);
}