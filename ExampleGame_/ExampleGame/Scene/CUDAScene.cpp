#include "CUDAScene.h"

#include <ExampleGame/CUDA/kernel.cuh>


LaiEngine::CUDAScene::CUDAScene(const SceneManager & sceneManager) : IGameScene(sceneManager)
{
}

LaiEngine::CUDAScene::~CUDAScene()
{

}

void LaiEngine::CUDAScene::Init()
{
	example.Init(nx, ny, 16, 16);

	constexpr size_t size_rgba = 4;
	const size_t buffer_size = nx * ny * size_rgba;

	textureBuffer = new uint8_t[buffer_size];

	constexpr int ns = 50;
	example.Update(textureBuffer, randBuffer, nx, ny, ns, 16, 16);
	
	this->image.create(nx, ny, sf::Color::White);
	this->texture.create(nx, ny);
	this->texture.update(textureBuffer);
	this->sprite.setTexture(this->texture);
}

void LaiEngine::CUDAScene::Update(const float dt)
{
}

void LaiEngine::CUDAScene::Release()
{
	example.Free();

	if (textureBuffer != nullptr)
	{
		delete[] textureBuffer;
	}
}

void LaiEngine::CUDAScene::Draw(std::weak_ptr<sf::RenderWindow> window)
{
	if (auto w = window.lock())
	{
		w->clear();
		w->draw(this->sprite);
	}
}

void LaiEngine::CUDAScene::InputProcess(std::weak_ptr<sf::RenderWindow> window, sf::Event & event)
{
}


bool LaiEngine::CUDAScene::KeyboardInput(std::weak_ptr<sf::RenderWindow> window)
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

bool LaiEngine::CUDAScene::MouseInput(std::weak_ptr<sf::RenderWindow> window)
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
