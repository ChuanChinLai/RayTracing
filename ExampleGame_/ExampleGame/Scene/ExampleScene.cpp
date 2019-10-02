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


GLuint textureColorbuffer = 0;

int tex_w = 1000, tex_h = 1000;

LaiEngine::ExampleScene::ExampleScene(const SceneManager & sceneManager) : IGameScene(sceneManager)
{

}

LaiEngine::ExampleScene::~ExampleScene()
{

}

void LaiEngine::ExampleScene::Init()
{
	Mesh mesh = Plane();
	mModel.Init(mesh);
	mScreen.Init(mesh);

	mComputeShader.UseProgram();
	mComputeShader.SetCameraPosition(mCamera.Position);
	mComputeShader.SetCameraLowerLeftCorner(mCamera.LowerLeftCorner);
	mComputeShader.SetCameraHorizontal(mCamera.Horizontal);
	mComputeShader.SetCameraVertical(mCamera.Vertical);



	{ // create the texture
		glGenTextures(1, &textureColorbuffer);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		// linear allows us to scale the window up retaining reasonable quality
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		// same internal format as compute shader input
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, tex_w, tex_h, 0, GL_RGBA, GL_FLOAT, NULL);
		// bind to image unit so can write to specific pixels from the shader
		glBindImageTexture(0, textureColorbuffer, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	}


	{ // query up the workgroups
		int work_grp_size[3], work_grp_inv;
		// maximum global work group (total work in a dispatch)
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_size[0]);
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_size[1]);
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_size[2]);
		printf("max global (total) work group size x:%i y:%i z:%i\n", work_grp_size[0], work_grp_size[1], work_grp_size[2]);

		// maximum local work group (one shader's slice)
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_size[0]);
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_size[1]);
		glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_size[2]);
		printf("max local (in one shader) work group sizes x:%i y:%i z:%i\n", work_grp_size[0], work_grp_size[1], work_grp_size[2]);

		// maximum compute shader invocations (x * y * z)
		glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv);
		printf("max computer shader invocations %i\n", work_grp_inv);
	}
}

void LaiEngine::ExampleScene::Update(const float dt)
{
	mCamera.Position += mCamera.Velocity * dt;
	mCamera.Velocity *= 0.95f;

	mCamera.Update();


	if (mCameraMoved)
	{
		mNumFrames = 0;
	}


	mComputeShader.SetNumFrames(mNumFrames);

	std::cout << mCamera.Position.x << std::endl;



	mNumFrames++;

	mCameraMoved = false;
}


void LaiEngine::ExampleScene::Release()
{

}

void LaiEngine::ExampleScene::Draw(std::weak_ptr<sf::RenderWindow> window)
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, tex_w, tex_h);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	mComputeShader.UseProgram();
	mComputeShader.SetCameraPosition(mCamera.Position);
	glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	mShader.UseProgram();
	mShader.SetViewMat(glm::mat4(1.0f));
	mShader.SetProjectedMat(glm::mat4(1.0f));


	mModel.BindVAO();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer);


	const GLsizei numElements = static_cast<GLsizei>(mModel.GetIndexCount());
	glDrawElements(GL_TRIANGLES, numElements, GL_UNSIGNED_INT, nullptr);
	glBindVertexArray(0);
}

void LaiEngine::ExampleScene::InputProcess(std::weak_ptr<sf::RenderWindow> window, sf::Event & event)
{
	keyboardInput(window);
	MouseInput(window);


	if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
	{
		mCameraMoved = true;
	}
	else
	{
		mCameraMoved = false;
	}
}

void LaiEngine::ExampleScene::Test()
{
	//std::clock_t start;
	//double duration;

	//start = std::clock();

	//int nx = 800;
	//int ny = 800;
	//int ns = 100;

	////image.create(nx, ny, sf::Color(0, 0, 0));

	//std::ofstream result("image.ppm");
	//if (!result.is_open())
	//	return;

	//result << "P3\n" << nx << " " << ny << " 255\n";

	//std::vector<GameObject*> list(4, nullptr);

	//list[0] = new Sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f, new Lambertian(glm::vec3(0.8f, 0.3f, 0.3f)));
	//list[1] = new Sphere(glm::vec3(0.0f, -100.5f, -1.0f), 100, new Lambertian(glm::vec3(0.8f, 0.8f, 0.0f)));
	//list[2] = new Sphere(glm::vec3(1.0f, 0.0f, -1.0f), 0.5f, new Metal(glm::vec3(0.8f, 0.6f, 0.2f), 0.5f));
	//list[3] = new Sphere(glm::vec3(-1.0f, 0.0f, -1.0f), 0.5f, new Metal(glm::vec3(0.8f, 0.8f, 0.8f), 0.5f));

	//GameObject* world = new GameObjectList(list);


	//for (int j = ny - 1; j >= 0; j--)
	//{
	//	for (int i = 0; i < nx; i++)
	//	{
	//		glm::vec3 color;

	//		for (int s = 0; s < ns; s++)
	//		{
	//			float u = static_cast<float>(i + Util::GetRandom()) / static_cast<float>(nx);
	//			float v = static_cast<float>(j + Util::GetRandom()) / static_cast<float>(ny);

	//			Ray r = camera.GetRay(u, v);
	//			color += Util::GetColor(r, world, 0);
	//		}

	//		color /= float(ns);
	//		color = glm::vec3(std::sqrt(color.x), std::sqrt(color.y), std::sqrt(color.z));

	//		int ir = int(255.99f * color.x);
	//		int ig = int(255.99f * color.y);
	//		int ib = int(255.99f * color.z);

	//		result << ir << " " << ig << " " << ib << "\n";

	//		//image.setPixel(i, ny - j - 1, sf::Color(ir, ig, ib));
	//	}
	//}

	//delete world;

	//result.close();

	//duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	//std::cout << "Rendering Time: " << duration << '\n';
}

void LaiEngine::ExampleScene::keyboardInput(std::weak_ptr<sf::RenderWindow> window)
{
	glm::vec3 dv;
	float speed = 0.5f;

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

	mCamera.Velocity += dv;
}

void LaiEngine::ExampleScene::MouseInput(std::weak_ptr<sf::RenderWindow> window)
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
}
