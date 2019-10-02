#include "SceneManager.h"
#include "IGameScene.h"

LaiEngine::SceneManager::SceneManager(const Engine& engine) : mEngine(engine), mCurrentScene(nullptr), mRunBegin(false)
{
}

LaiEngine::SceneManager::~SceneManager()
{

}

void LaiEngine::SceneManager::SetGameScene(std::shared_ptr<IGameScene> scene)
{
	mRunBegin = false;

	if (mCurrentScene != nullptr)
	{
		mCurrentScene->Release();
	}

	mCurrentScene = scene;
}

const LaiEngine::IGameScene& LaiEngine::SceneManager::GetGameScene()
{
	return *mCurrentScene;
}

void LaiEngine::SceneManager::Update(const float dt)
{
	if (mRunBegin == false)
	{
		mCurrentScene->Init();
		mRunBegin = true;
	}

	mCurrentScene->Update(dt);
}

void LaiEngine::SceneManager::Release()
{
	mCurrentScene->Release();
}

void LaiEngine::SceneManager::Draw(std::weak_ptr<sf::RenderWindow> window)
{
	mCurrentScene->Draw(window);
}

void LaiEngine::SceneManager::InputProcess(std::weak_ptr<sf::RenderWindow> window, sf::Event& event)
{
	if (mCurrentScene != nullptr)
	{
		mCurrentScene->InputProcess(window, event);
	}
}
