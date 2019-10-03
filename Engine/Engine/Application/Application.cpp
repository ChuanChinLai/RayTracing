#include "Application.h"

#include <Engine/Scene/SceneManager.h>

#include <GL/glew.h>

#include <SFML/Graphics.hpp>
#include <iostream>

LaiEngine::Engine::Engine(const std::string& title) : mRenderWindow(std::make_shared<sf::RenderWindow>()), mSceneManager(std::make_shared<SceneManager>(*this))
{
	try
	{
		if (!InitGL(title))
		{
			throw std::runtime_error("Failed to initialize OpenGL");
		}
	}
	catch (std::runtime_error& error)
	{
		std::cout << error.what() << std::endl;
	}
}

LaiEngine::Engine::~Engine()
{

}

void LaiEngine::Engine::GameLoop()
{
	mIsGameRunning = mRenderWindow->isOpen();

	sf::Clock clock;
	sf::Time accumulator = sf::Time::Zero;
	sf::Time ups = sf::seconds(1.f / 60.f);


	while (mIsGameRunning)
	{
		while (accumulator > ups)
		{
			mSceneManager->Update(ups.asSeconds());
			accumulator -= ups;
		}

		sf::Event event;
		mSceneManager->InputProcess(mRenderWindow, event);

		mSceneManager->Draw(mRenderWindow);

		mRenderWindow->display();

		HandleEvents(event);

		accumulator += clock.restart();
	}
}

bool LaiEngine::Engine::InitGL(const std::string & title)
{
	sf::ContextSettings settings;
	settings.antialiasingLevel = 0;
	settings.majorVersion = 4;
	settings.minorVersion = 3;
	settings.depthBits = 24;
	settings.stencilBits = 8;

	try
	{
		sf::String sf_title(title.c_str());
		mRenderWindow->create({ 800, 800 }, sf_title, sf::Style::Close, settings);

		GLenum error = glewInit();
		if (GLEW_OK != error)
		{
			throw std::runtime_error("Failed to load OpenGL");
		}
	}
	catch (std::runtime_error& error)
	{
		std::cout << error.what() << std::endl;
		return false;
	}


	glViewport(0, 0, mRenderWindow->getSize().x, mRenderWindow->getSize().y);

	glEnable(GL_DEPTH_TEST);

	return true;
}
