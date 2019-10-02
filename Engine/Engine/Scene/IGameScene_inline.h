#pragma once

#include "IGameScene.h"
#include "SceneManager.h"

namespace LaiEngine
{
	inline IGameScene::IGameScene(const SceneManager& sceneManager) : mSceneManager(sceneManager)
	{

	}

	inline IGameScene::~IGameScene()
	{

	}
}