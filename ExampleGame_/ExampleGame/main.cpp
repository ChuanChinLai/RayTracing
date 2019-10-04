#include <iostream>
#include <Core/GameEngine.h>

#include "cuda_runtime.h"

int main()
{
	//int dev = 0;
	//cudaDeviceProp devProp;
	//cudaGetDeviceProperties(&devProp, dev);
	//std::cout << "GPU device " << dev << ": " << devProp.name << std::endl;
	//std::cout << "SM:" << devProp.multiProcessorCount << std::endl;
	//std::cout << "" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
	//std::cout << "" << devProp.maxThreadsPerBlock << std::endl;
	//std::cout << "" << devProp.maxThreadsPerMultiProcessor << std::endl;
	//std::cout << "" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

	//clock_t start, end;

	//double *a, *b, *c;
	//int size = N * sizeof(double);

	//a = (double *)malloc(size);
	//b = (double *)malloc(size);
	//c = (double *)malloc(size);

	//for (int i = 0; i < N; i++)
	//{
	//	a[i] = b[i] = i;
	//	c[i] = 0;
	//}

	//start = clock();
	//serial_add(a, b, c, N, M);

	//printf("c[%d] = %f\n", 0, c[0]);
	//printf("c[%d] = %f\n", N - 1, c[N - 1]);

	//end = clock();

	//float time1 = ((float)(end - start)) / CLOCKS_PER_SEC;
	//printf("CPU: %f seconds\n", time1);

	//start = clock();

	//addWithCuda(a, b, c, size);

	//end = clock();
	//float time2 = ((float)(end - start)) / CLOCKS_PER_SEC;
	//printf("CUDA: %f seconds, Speedup: %f\n", time2, time1 / time2);

	//free(a);
	//free(b);
	//free(c);

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