#pragma once
#pragma once

#include <Engine/Utility/NonCopyable.h>
#include <Engine/Utility/NonMovable.h>


#include <string>
#include <memory>

namespace sf
{
	class RenderWindow;
	class Event;
}

namespace LaiEngine
{
	class SceneManager;

	class Engine : public NonMovable, public NonCopyable
	{
	public:

		Engine(const std::string& title);
		virtual ~Engine();

		virtual bool Init() = 0;
		virtual void GameLoop();
		virtual void Release() = 0;

	protected:

		virtual void HandleEvents(sf::Event& event) = 0;
		virtual bool InitGL(const std::string& title);

		std::shared_ptr<sf::RenderWindow> mRenderWindow;
		std::shared_ptr<SceneManager> mSceneManager;

		bool   mIsGameRunning = false;
	};
}