#pragma once

#include <Engine/Engine.h>

namespace LaiEngine
{
	class GameEngine : public Engine
	{
	public:

		GameEngine(const std::string& title);

		bool Init() override;
		void Release() override;

	private:

		void HandleEvents(sf::Event& event) override;
	};
}