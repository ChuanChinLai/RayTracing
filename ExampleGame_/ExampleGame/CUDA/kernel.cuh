#include <cstdint>


namespace LaiEngine
{
	namespace CUDA
	{
		void Init(int nx, int ny, int tx, int ty);
		void Update(uint8_t* buffer, const size_t buffer_size, int nx, int ny, int tx, int ty);
	}
}