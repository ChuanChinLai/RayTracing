#version 430

layout (local_size_x = 1, local_size_y = 1) in;
layout (rgba32f, binding = 0) uniform image2D img_output;


uint g_state = 0;

//Constants: 

#define MATERIAL_LAMBERTIAN 0
#define MATERIAL_METAL 1
#define MATERIAL_DIELECTRIC 2


struct Ray
{
    vec3 Origin;
    vec3 Direction;
};

struct Sphere
{
	vec3 Position;
	float Radius;
	int materialID;
};


struct ShadeRec
{
	float t;
	int materialID;
	vec3 Point;
	vec3 Normal;
};


struct Metal
{
    float roughness;
};

struct Dielectric
{
    float ref_idx;
};

struct Material
{
    int type;
    vec3 albedo;
    Metal metal;
    Dielectric dielectric;
};


struct Scene
{   
    int num_spheres;
	int num_materials;
    Sphere spheres[32];
	Material materials[32];
};


uniform int numFrames;
uniform int getInputs;

uniform mat4 inverseViewMat;
uniform mat4 inverseProjectedMat;


uint rand(inout uint state)
{
    uint x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 15;
    state = x;
    return x;
}

float RandomFloat_01(inout uint state)
{
    return (rand(state) & 0xFFFFFF) / 16777216.0f;
}


vec3 GetRandomInUnitShpere(inout uint state)
{
    float z = RandomFloat_01(state) * 2.0f - 1.0f;
    float t = RandomFloat_01(state) * 2.0f * 3.1415926f;
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float x = r * cos(t);
    float y = r * sin(t);
    vec3 res = vec3(x, y, z);
    res *= pow(RandomFloat_01(state), 1.0f / 3.0f);
    return res;
}


//Physics

vec3 Reflect(vec3 v, vec3 n)
{
	return v - 2.0f * dot(v, n) * n;
}

bool Refract(in vec3 v, in vec3 n, in float ni_over_nt, out vec3 refracted)
{
    vec3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);

    if (discriminant > 0)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }

	return false;
}


float Schlick(float cosine, float ref_idx)
{
    float r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5);
}

//Ray


Ray GetRay(float u, float v)
{
    u = u * 2.0 - 1.0;
    v = v * 2.0 - 1.0;
	
	vec4 clip_pos = vec4(u, v, -1.0, 1.0);
    vec4 view_pos = inverseProjectedMat * clip_pos;
	
	vec3 dir = vec3(inverseViewMat * vec4(view_pos.x, view_pos.y, -1.0, 0.0));
    dir = normalize(dir);
	
	vec4 origin = inverseViewMat * vec4(0.0, 0.0, 0.0, 1.0);
    origin.xyz /= origin.w;
	
	
	Ray ray; 

	ray.Origin = origin.xyz; 
	ray.Direction = dir;
	
	return ray;
}


bool ScatterLambertian(in Ray ray, in ShadeRec rec, in Material mat, out vec3 attenuation, out Ray scattered)
{
    vec3 new_dir = rec.Point + rec.Normal + GetRandomInUnitShpere(g_state);
	
	scattered.Origin = rec.Point; 
	scattered.Direction = normalize(new_dir - rec.Point);
	
    attenuation = mat.albedo;

    return true;
}


bool ScatterMetal(in Ray ray, in ShadeRec rec, in Material mat, out vec3 attenuation, out Ray scattered)
{
    vec3 reflected = Reflect(ray.Direction, rec.Normal);
	
	scattered.Origin = rec.Point; 
	scattered.Direction = normalize(reflected + mat.metal.roughness * GetRandomInUnitShpere(g_state));
	
    attenuation = mat.albedo;

    return dot(scattered.Direction, rec.Normal) > 0;
}


bool ScatterDielectric(in Ray ray, in ShadeRec rec, in Material mat, out vec3 attenuation, out Ray scattered)
{
	vec3 outward_normal;
    vec3 reflected = Reflect(ray.Direction, rec.Normal);
    float ni_over_nt;
    attenuation = vec3(1.0, 1.0, 1.0);
    vec3 refracted;
    float reflect_prob;
    float cosine;
	
	if (dot(ray.Direction, rec.Normal) > 0)
    {
        outward_normal = -rec.Normal;
        ni_over_nt = mat.dielectric.ref_idx;
        cosine = mat.dielectric.ref_idx * dot(ray.Direction, rec.Normal) / length(ray.Direction);
    }
    else
    {
        outward_normal = rec.Normal;
        ni_over_nt = 1.0 / mat.dielectric.ref_idx;
        cosine = -dot(ray.Direction, rec.Normal) / length(ray.Direction);
    }

    if (Refract(ray.Direction, outward_normal, ni_over_nt, refracted))
	{
        reflect_prob = Schlick(cosine, mat.dielectric.ref_idx);
	}
    else
	{
        reflect_prob = 1.0;
	}
	
	
    if (RandomFloat_01(g_state) < reflect_prob)
	{	
		scattered.Origin = rec.Point; 
		scattered.Direction = reflected;
	}
    else
	{
		scattered.Origin = rec.Point; 
		scattered.Direction = refracted;
	}


    return true;
}



bool HitSphere(Ray ray, Sphere sphere, float t_min, float t_max, out ShadeRec rec)
{
	vec3 r = ray.Origin - sphere.Position;      
	float a = dot(ray.Direction, ray.Direction);	
	float b = dot(r, ray.Direction);                                                
	float c = dot(r, r) - sphere.Radius * sphere.Radius;                          
	float discriminant = b * b - a * c;    

	if(discriminant > 0.0f)
	{
		
		float temp = (-b - sqrt(discriminant)) / a;

		if (temp < t_max && temp > t_min) 
		{
			rec.t = temp;
			rec.Point = ray.Origin + rec.t * ray.Direction;
			rec.Normal = normalize((rec.Point - sphere.Position) / sphere.Radius);
			rec.materialID = sphere.materialID;
			
			return true;
		}
		
		temp = (-b + sqrt(discriminant)) / a;

		if (temp < t_max && temp > t_min) 
		{
			rec.t = temp;
			rec.Point = ray.Origin + rec.t * ray.Direction;
			rec.Normal = normalize((rec.Point - sphere.Position) / sphere.Radius);
			rec.materialID = sphere.materialID;
			
			return true;
		}
		
	}
	
	
	return false;
}


bool HitTheScene(Ray ray, float t_min, float t_max, in Scene scene, out ShadeRec rec)
{
	float closest_so_far = t_max;
	bool hitAnything = false;

	
	for(int i = 0; i < scene.num_spheres; i++)
	{	
		if(HitSphere(ray, scene.spheres[i], t_min, closest_so_far, rec))
		{
			hitAnything = true; 
			closest_so_far = rec.t;
		}
	}
	
	return hitAnything; 
}



vec3 GetColor(Ray ray, in Scene scene)
{
	ShadeRec rec;

	int depth_max = 50; 
	int depth = 0; 
	
	vec3 color = vec3(1.0, 1.0, 1.0);
	vec3 attenuation = vec3(0.0, 0.0, 0.0);

	
	if(getInputs == 1)
	{
		depth_max = 10;
	}

	

	while(depth < depth_max)
	{
		if(HitTheScene(ray, 0.001f, 10000.0f, scene, rec))
		{
		
			Ray scattered;
			
			if(scene.materials[rec.materialID].type == MATERIAL_LAMBERTIAN)
			{
				if(ScatterLambertian(ray, rec, scene.materials[rec.materialID], attenuation, scattered))
				{
					color *= attenuation; 
					ray = scattered;
				}
				else
				{
					attenuation = vec3(0.0, 0.0, 0.0);
					color += attenuation;
					break;
				}
			}
			else if(scene.materials[rec.materialID].type == MATERIAL_METAL)
			{
				if(ScatterMetal(ray, rec, scene.materials[rec.materialID], attenuation, scattered))
				{
					color *= attenuation; 
					ray = scattered;
				}
				else
				{
					attenuation = vec3(0.0, 0.0, 0.0);
					color += attenuation;
					break;
				}
			}
			else if(scene.materials[rec.materialID].type == MATERIAL_DIELECTRIC)
			{
				if (ScatterDielectric(ray, rec, scene.materials[rec.materialID], attenuation, scattered))
                {
                    color *= attenuation;
                    ray = scattered;
                }
                else
                {
                    attenuation = vec3(0.0, 0.0, 0.0);
                    color *= attenuation;
                    break;
                }
			}
		}
		else
		{
			vec3 unit_dir = normalize(ray.Direction);
			float t = 0.5f * (unit_dir.y + 1.0f);
			vec3 backgroundColor = (1.0 - t) * vec3(1, 1, 1) + t * vec3(0.5, 0.7, 1.0);
			
			color *= backgroundColor;
			
			break;
		}
		
		depth++;
	}
	
    if (depth > depth_max) 
	    return vec3(0.0, 0.0, 0.0);	
	
    return color;
    
}


void main () 
{                                  
	g_state = gl_GlobalInvocationID.x * 1973 + gl_GlobalInvocationID.y * 9277 + uint(numFrames) * 2699 | 1;
	
	Scene scene;

    scene.num_spheres = 5;
	
	
	// Blue Lambertian
	scene.materials[0].type = MATERIAL_LAMBERTIAN;
    scene.materials[0].albedo = vec3(0.8, 0.8, 0.0);
	
    // Floor
    scene.materials[1].type = MATERIAL_LAMBERTIAN;
    scene.materials[1].albedo = vec3(0.8, 0.3, 0.3);
	
	scene.materials[2].type = MATERIAL_METAL;
    scene.materials[2].albedo = vec3(0.8, 0.6, 0.2);
    scene.materials[2].metal.roughness = 0.3;
	
	scene.materials[3].type = MATERIAL_METAL;
    scene.materials[3].albedo = vec3(0.8, 0.8, 0.8);
    scene.materials[3].metal.roughness = 0.0;
	
	scene.materials[4].type = MATERIAL_DIELECTRIC;
    scene.materials[4].albedo = vec3(0.8, 0.8, 0.8);
    scene.materials[4].dielectric.ref_idx = 1.5;
	
	
	
	scene.spheres[0].Position = vec3 (0.0f, 0.0f, -1.0f); 
	scene.spheres[0].Radius = 0.5f;
	scene.spheres[0].materialID = 1;
	
	scene.spheres[1].Position = vec3 (0.0f, -100.5f, -1.0f);                                       
	scene.spheres[1].Radius = 100.0f; 
	scene.spheres[1].materialID = 0;
	
	scene.spheres[2].Position = vec3(-1.0, 0.0, -1.0);
    scene.spheres[2].Radius = 0.5f;
    scene.spheres[2].materialID = 4;
	
	
	scene.spheres[3].Position = vec3(1.0, 0.0, -1.0);
    scene.spheres[3].Radius = 0.5f;
    scene.spheres[3].materialID = 2;
	
	scene.spheres[4].Position = vec3(-1.0, 0.0, -1.0);
    scene.spheres[4].Radius = -0.45f;
    scene.spheres[4].materialID = 4;
	
	
	
	
	ivec2 pixel_coords = ivec2 (gl_GlobalInvocationID.xy);                                                        
	ivec2 dims = imageSize (img_output);            
	
	vec3 color = vec3 (0.0);
	
	int ns = 5;
	
	if(getInputs == 1)
	{
		ns = 2;
	}
	
	
	for(int i = 0; i < ns; i++)
	{
		float u = (float(pixel_coords.x + RandomFloat_01(g_state)) / dims.x);                 
		float v = (float(pixel_coords.y + RandomFloat_01(g_state)) / dims.y);
		
		Ray ray = GetRay(u, v);
		
		color += GetColor(ray, scene);
	}
	
	color /= float(ns);
	
	color = vec3(sqrt(color.x), sqrt(color.y), sqrt(color.z));
	
	vec3 prev_color = imageLoad(img_output, pixel_coords).rgb;
	vec3 final = mix(color, prev_color, float(numFrames) / float(numFrames + 1));
	
	imageStore (img_output, pixel_coords, vec4(final, 1.0f));                          
};