#include "ExampleScene.h"

#include <Engine/Engine.h>
#include <Engine/Utility/FileReader.h>

#include <GameObject/Sphere.h>

#include <Core/Camera.h>
#include <Core/Ray.h>
#include <Core/Utility.h>

#include <Material/Material.h>

#include <GL/glew.h>

#include <iostream>
#include <fstream>
#include <random>

extern void GetPixelColorWithCuda(uint8_t* buffer, const size_t buffer_size, int nx, int ny, int tx, int ty);

GLuint textureColorbuffer = 0;

int tex_w = 600, tex_h = 600;

sf::Image image;
sf::Sprite sprite;
sf::Texture texture;

std::ostream& operator<<(std::ostream& os, const glm::vec3& v)
{
	os << v.x << '/' << v.y << '/' << v.z;
	return os;
}

LaiEngine::ExampleScene::ExampleScene(const SceneManager & sceneManager) : IGameScene(sceneManager)
{

}

LaiEngine::ExampleScene::~ExampleScene()
{

}

void LaiEngine::ExampleScene::Init()
{
	Test();

	//Mesh mesh = Plane();
	//mModel.Init(mesh);

	//{ // create the texture
	//	glGenTextures(1, &textureColorbuffer);
	//	glActiveTexture(GL_TEXTURE0);
	//	glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	//	// linear allows us to scale the window up retaining reasonable quality
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//	// same internal format as compute shader input
	//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, tex_w, tex_h, 0, GL_RGBA, GL_FLOAT, NULL);
	//	// bind to image unit so can write to specific pixels from the shader
	//	glBindImageTexture(0, textureColorbuffer, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	//}


	//{ // query up the workgroups
	//	int work_grp_size[3], work_grp_inv;
	//	// maximum global work group (total work in a dispatch)
	//	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_size[0]);
	//	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_size[1]);
	//	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_size[2]);
	//	printf("max global (total) work group size x:%i y:%i z:%i\n", work_grp_size[0], work_grp_size[1], work_grp_size[2]);

	//	// maximum local work group (one shader's slice)
	//	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0]);
	//	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1]);
	//	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2]);
	//	printf("max local (in one shader) work group sizes x:%i y:%i z:%i\n", work_grp_size[0], work_grp_size[1], work_grp_size[2]);

	//	// maximum compute shader invocations (x * y * z)
	//	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv);
	//	printf("max computer shader invocations %i\n", work_grp_inv);
	//}
}

void LaiEngine::ExampleScene::Update(const float dt)
{
	//mCamera.Update(dt);

	//mComputeShader.UseProgram();
	//mComputeShader.SetNumFrames(mNumFrames);
	//mComputeShader.SetGetInputs(mGetInputs);
	//mComputeShader.SetMixingRatio(static_cast<float>(mNumFrames) / static_cast<float>(mNumFrames + 1));

	//mComputeShader.SetInverseViewMat(glm::inverse(mCamera.GetViewMatrix()));
	//mComputeShader.SetInverseProjectedMat(glm::inverse(mCamera.GetProjectedMatrix()));

	//if (mGetInputs)
	//{
	//	mNumFrames = 0;
	//}

	////std::cout << mGetInputs << " " << mNumFrames << std::endl;

	//mGetInputs = false;

	//mNumFrames++;
}


void LaiEngine::ExampleScene::Release()
{

}

void LaiEngine::ExampleScene::Draw(std::weak_ptr<sf::RenderWindow> window)
{
	//glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	window.lock()->clear();

	window.lock()->draw(sprite);

	window.lock()->display();

	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//mComputeShader.UseProgram();
	//glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);
	//glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	//mShader.UseProgram();
	////mShader.SetViewMat(glm::mat4(1.0f));
	////mShader.SetProjectedMat(glm::mat4(1.0f));

	//mModel.BindVAO();

	//const GLsizei numElements = static_cast<GLsizei>(mModel.GetIndexCount());
	//glDrawElements(GL_TRIANGLES, numElements, GL_UNSIGNED_INT, nullptr);
}

void LaiEngine::ExampleScene::InputProcess(std::weak_ptr<sf::RenderWindow> window, sf::Event & event)
{
//	KeyboardInput(window); 
//	MouseInput(window);
}

void LaiEngine::ExampleScene::Test()
{
	int nx = 800; 
	int ny = 800;

	constexpr size_t rgba = 4;
	const size_t buffer_size = nx * ny * 4;
	//float* buffer = (float*)malloc(buffer_size * sizeof(float));

	uint8_t* buffer = (uint8_t*)malloc(buffer_size);
	GetPixelColorWithCuda(buffer, buffer_size, nx, ny, 8, 8);

	image.create(nx, ny, sf::Color::Blue);

	//std::vector<GameObject*> list(4, nullptr);

	//list[0] = new Sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f, new Lambertian(glm::vec3(0.8f, 0.3f, 0.3f)));
	//list[1] = new Sphere(glm::vec3(0.0f, -100.5f, -1.0f), 100, new Lambertian(glm::vec3(0.8f, 0.8f, 0.0f)));
	//list[2] = new Sphere(glm::vec3(1.0f, 0.0f, -1.0f), 0.5f, new Metal(glm::vec3(0.8f, 0.6f, 0.2f), 0.5f));
	//list[3] = new Sphere(glm::vec3(-1.0f, 0.0f, -1.0f), 0.5f, new Metal(glm::vec3(0.8f, 0.8f, 0.8f), 0.5f));

	//GameObject* world = new GameObjectList(list);

	//int index = 0;

	clock_t start, stop;
	start = clock();

	//std::ofstream result("image1.ppm");
	//if (!result.is_open())
	//	return;

	//result << "P3\n" << nx << " " << ny << " 255\n";

	//int index = 0;

	//for (int j = ny - 1; j >= 0; j--)
	//{
	//	for (int i = 0; i < nx; i++)
	//	{
	//		size_t pixel_index = j * 4 * nx + i * 4;

	//		float r = buffer[pixel_index + 0];
	//		float g = buffer[pixel_index + 1];
	//		float b = buffer[pixel_index + 2];

	//		int ir = int(255.99f * r);
	//		int ig = int(255.99f * g);
	//		int ib = int(255.99f * b);

	//		buffer[index + 0] = ir;
	//		buffer[index + 1] = ig;
	//		buffer[index + 2] = ib;
	//		buffer[index + 3] = 255;

	//		index += 4;

	//		result << ir << " " << ig << " " << ib << "\n";
	//		//std::cout << ir << " " << ig << " " << ib << "\n";
	//	}
	//}

	//result.close();

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	//for (int j = ny - 1; j >= 0; j--)
	//{
	//	for (int i = 0; i < nx; i++)
	//	{
	//		glm::vec3 color;

	//		for (int s = 0; s < ns; s++)
	//		{
	//			float u = static_cast<float>(i + Util::GetRandom()) / static_cast<float>(nx);
	//			float v = static_cast<float>(j + Util::GetRandom()) / static_cast<float>(ny);

	//			Ray r = mCamera.GetRay(u, v);
	//			color += Util::GetColor(r, world, 0);
	//		}

	//		color /= float(ns);
	//		color = glm::vec3(std::sqrt(color.x), std::sqrt(color.y), std::sqrt(color.z));

	//		uint8_t ir = static_cast<uint8_t>(255.99f * color.x);
	//		uint8_t ig = static_cast<uint8_t>(255.99f * color.y);
	//		uint8_t ib = static_cast<uint8_t>(255.99f * color.z);

	//		result << ir << " " << ig << " " << ib << "\n";

	//		image.setPixel(i, ny - j - 1, sf::Color(ir, ig, ib));

	//		pixels[index + 0] = ir;
	//		pixels[index + 1] = ig;
	//		pixels[index + 2] = ib;
	//		pixels[index + 3] = 255;

	//		index += 4;
	//	}
	//}


	//texture.loadFromImage(image);

	texture.create(nx, ny);
	texture.update(buffer);

	sprite.setTexture(texture);


	//delete world;
}

bool LaiEngine::ExampleScene::KeyboardInput(std::weak_ptr<sf::RenderWindow> window)
{
	glm::vec3 dv;
	constexpr float speed = 0.5f;

	if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
	{
		dv.x = glm::cos(glm::radians(mCamera.Rotation.y + 90)) * speed;
		dv.z = glm::sin(glm::radians(mCamera.Rotation.y + 90)) * speed;
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
	{
		dv.x = -glm::cos(glm::radians(mCamera.Rotation.y + 90)) * speed;
		dv.z = -glm::sin(glm::radians(mCamera.Rotation.y + 90)) * speed;
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
	{
		dv.x = glm::cos(glm::radians(mCamera.Rotation.y)) * speed;
		dv.z = glm::sin(glm::radians(mCamera.Rotation.y)) * speed;
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
	{
		dv.x = -glm::cos(glm::radians(mCamera.Rotation.y)) * speed;
		dv.z = -glm::sin(glm::radians(mCamera.Rotation.y)) * speed;
	}

	if (glm::length(dv) > 0.0f)
	{
		mGetInputs = true;
	}

	mCamera.Velocity += dv;

	return mGetInputs;
}

bool LaiEngine::ExampleScene::MouseInput(std::weak_ptr<sf::RenderWindow> window)
{
	static auto const BOUND = 80;
	static auto lastMousePosition = sf::Mouse::getPosition(*window.lock());
	auto change = sf::Mouse::getPosition() - lastMousePosition;

	mCamera.Rotation.y += static_cast<float>(change.x * 0.05);
	mCamera.Rotation.x += static_cast<float>(change.y * 0.05);

	if (mCamera.Rotation.x > BOUND) mCamera.Rotation.x = BOUND;
	else if (mCamera.Rotation.x < -BOUND) mCamera.Rotation.x = -BOUND;

	if (mCamera.Rotation.y > 360) mCamera.Rotation.y = 0;
	else if (mCamera.Rotation.y < 0)    mCamera.Rotation.y = 360;

	auto cx = static_cast<int>(window.lock()->getSize().x / 2);
	auto cy = static_cast<int>(window.lock()->getSize().y / 2);

	sf::Mouse::setPosition({ cx, cy }, *window.lock());

	lastMousePosition = sf::Mouse::getPosition();

	return 	mGetInputs = (change.x == 0 && change.y == 0 && mGetInputs == false) ? false : true;
}
