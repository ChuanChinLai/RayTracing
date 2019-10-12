#include "ExampleScene.h"

#include <Engine/Engine.h>
#include <Engine/Utility/FileReader.h>

#include <GameObject/Sphere.h>

#include <Core/Camera.h>
#include <Core/Ray.h>

#include <Material/Material.h>

#include <GL/glew.h>

#include <iostream>
#include <fstream>
#include <random>

GLuint textureColorbuffer = 0;

int tex_w = 600, tex_h = 600;


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
	Mesh mesh = Plane();
	mModel.Init(mesh);

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
	mCamera.Update(dt);

	mComputeShader.UseProgram();
	mComputeShader.SetNumFrames(mNumFrames);
	mComputeShader.SetGetInputs(mGetInputs);
	mComputeShader.SetMixingRatio(static_cast<float>(mNumFrames) / static_cast<float>(mNumFrames + 1));

	mComputeShader.SetInverseViewMat(glm::inverse(mCamera.GetViewMatrix()));
	mComputeShader.SetInverseProjectedMat(glm::inverse(mCamera.GetProjectedMatrix()));

	if (mGetInputs)
	{
		mNumFrames = 0;
	}

	//std::cout << mGetInputs << " " << mNumFrames << std::endl;

	mGetInputs = false;

	mNumFrames++;
}


void LaiEngine::ExampleScene::Release()
{

}

void LaiEngine::ExampleScene::Draw(std::weak_ptr<sf::RenderWindow> window)
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	mComputeShader.UseProgram();
	glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	mShader.UseProgram();
	//mShader.SetViewMat(glm::mat4(1.0f));
	//mShader.SetProjectedMat(glm::mat4(1.0f));
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
	mModel.BindVAO();

	const GLsizei numElements = static_cast<GLsizei>(mModel.GetIndexCount());
	glDrawElements(GL_TRIANGLES, numElements, GL_UNSIGNED_INT, nullptr);
}

void LaiEngine::ExampleScene::InputProcess(std::weak_ptr<sf::RenderWindow> window, sf::Event & event)
{
	KeyboardInput(window); 
	MouseInput(window);
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
