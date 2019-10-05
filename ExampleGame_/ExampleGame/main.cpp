#include <iostream>
#include <Core/GameEngine.h>

int main()
{
	{
		LaiEngine::GameEngine game("Ray Tracing");

		if (!game.Init())
			return -1;

		game.GameLoop();
		game.Release();
	}


#if defined _DEBUG
	_CrtDumpMemoryLeaks();
#endif // _DEBUG

	system("pause");

	return 0;
}