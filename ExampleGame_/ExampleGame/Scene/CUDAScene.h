#pragma once

#include <Engine/Scene/IGameScene.h>
#include <Engine/Model/Model.h>
#include <Core/Camera.h>

#include <ExampleGame/CUDA/CUDAExample.h>

#include <CUDA/curand.h>
#include <CUDA/curand_kernel.h>
#include <SFML/Graphics.hpp>


namespace LaiEngine
{
	class CUDAScene : public IGameScene
	{
	public:

		CUDAScene(const SceneManager& sceneManager);
		~CUDAScene();

		void Init()	override;
		void Update(const float dt) override;
		void Release() override;

		void Draw(std::weak_ptr<sf::RenderWindow> window) override;
		void InputProcess(std::weak_ptr<sf::RenderWindow> window, sf::Event& event) override;

	private:

		bool KeyboardInput(std::weak_ptr<sf::RenderWindow> window);
		bool MouseInput(std::weak_ptr<sf::RenderWindow> window);


		CUDA::CUDAExample example;


		Camera mCamera;

		CUDA::Camera cc;

		int nx = 600;
		int ny = 600;

		uint8_t* textureBuffer = nullptr;
		curandState* randBuffer = nullptr;


		sf::Image image;
		sf::Sprite sprite;
		sf::Texture texture;

		bool mGetInputs = false;
	};
}