#include "GameEngine.h"

#include <Scene/CUDAScene.h>
#include <Scene/ExampleScene.h>

#include <iostream>


LaiEngine::GameEngine::GameEngine(const std::string & title) : Engine(title)
{

}

bool LaiEngine::GameEngine::Init()
{
	std::shared_ptr<CUDAScene> scene = std::make_shared<CUDAScene>(*mSceneManager);
	mSceneManager->SetGameScene(scene);

	return true;
}


void LaiEngine::GameEngine::Release()
{
}


void LaiEngine::GameEngine::HandleEvents(sf::Event & event)
{
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
	{
		mIsGameRunning = false;
	}

	while (mRenderWindow->pollEvent(event))
	{
		if (event.type == sf::Event::Closed)
		{
			mIsGameRunning = false;
		}
		else
		{
			mSceneManager->InputProcess(mRenderWindow, event);
		}
	}
}
