#pragma once

#include <Engine/Utility/NonCopyable.h>
#include <Engine/Utility/NonMovable.h>

#include <Engine/Engine.h>

#include <memory>

namespace sf
{
	class RenderWindow;
	class Event;
}

namespace LaiEngine
{
	class IGameScene;

	class SceneManager : public NonMovable, public NonCopyable
	{
	public:

		SceneManager(const Engine& engine);
		~SceneManager();

		void SetGameScene(std::shared_ptr<IGameScene> scene);
		const IGameScene& GetGameScene();

		void Update(const float dt);
		void Release();

		void Draw(std::weak_ptr<sf::RenderWindow> window);
		void InputProcess(std::weak_ptr<sf::RenderWindow> window, sf::Event& event);

	private:

		std::shared_ptr<IGameScene>       mCurrentScene;
		bool 							  mRunBegin;

		const Engine&                     mEngine;
	};
}