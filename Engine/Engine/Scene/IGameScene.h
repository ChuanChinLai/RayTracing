#pragma once

#include <Engine/Utility/NonCopyable.h>
#include <Engine/Utility/NonMovable.h>

#include <memory>

namespace sf
{
	class RenderWindow;
	class Event;
}

namespace LaiEngine
{
	class SceneManager;

	class IGameScene : public NonMovable, public NonCopyable
	{
	public:

		IGameScene(const SceneManager& sceneManager);
		virtual ~IGameScene();

		virtual void Init() = 0;
		virtual void Update(const float dt) = 0;
		virtual void Release() = 0;
		
		virtual void Draw(std::weak_ptr<sf::RenderWindow> window) = 0;
		virtual void InputProcess(std::weak_ptr<sf::RenderWindow> window, sf::Event& event) = 0;

	protected:

		const SceneManager&	mSceneManager;
	};
}

#include "IGameScene_inline.h"