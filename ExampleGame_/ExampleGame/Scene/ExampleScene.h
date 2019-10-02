#pragma once

#include <Engine/Scene/IGameScene.h>
#include <Engine/Model/Model.h>

#include <Core/Camera.h>

#include <ExampleGame/Shader/BasicShader.h>
#include <ExampleGame/Shader/ComputeShader.h>

namespace LaiEngine
{
	class ExampleScene : public IGameScene
	{
	public:

		ExampleScene(const SceneManager& sceneManager);
		~ExampleScene();

		void Init()	override;
		void Update(const float dt) override;
		void Release() override;

		void Draw(std::weak_ptr<sf::RenderWindow> window) override;
		void InputProcess(std::weak_ptr<sf::RenderWindow> window, sf::Event& event) override;

	private:

		void Test();

		void keyboardInput(std::weak_ptr<sf::RenderWindow> window);
		void MouseInput(std::weak_ptr<sf::RenderWindow> window);


		Model mModel;
		Camera mCamera;

		BasicShader mShader;
		ComputeShader mComputeShader;

		int mNumFrames = 0;
		bool mCameraMoved = false;

	};
}