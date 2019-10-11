#include "CUDAScene.h"

LaiEngine::CUDAScene::CUDAScene(const SceneManager & sceneManager) : IGameScene(sceneManager)
{
}

LaiEngine::CUDAScene::~CUDAScene()
{
}

void LaiEngine::CUDAScene::Init()
{
	int nx = 500;
	int ny = 500;

	constexpr size_t size_rgba = 4;
	const size_t buffer_size = nx * ny * size_rgba;

	uint8_t* buffer = new uint8_t[buffer_size];
	//RenderWithCuda(buffer, buffer_size, nx, ny, 16, 16);

	this->image.create(nx, ny, sf::Color::White);
	this->texture.create(nx, ny);
	this->texture.update(buffer);
	this->sprite.setTexture(this->texture);


	delete[] buffer;
}

void LaiEngine::CUDAScene::Update(const float dt)
{
}

void LaiEngine::CUDAScene::Release()
{
}

void LaiEngine::CUDAScene::Draw(std::weak_ptr<sf::RenderWindow> window)
{
	if (auto w = window.lock())
	{
		w->clear();
		w->draw(this->sprite);
		w->display();
	}
}

void LaiEngine::CUDAScene::InputProcess(std::weak_ptr<sf::RenderWindow> window, sf::Event & event)
{
}
