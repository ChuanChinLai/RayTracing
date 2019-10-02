#pragma once

namespace LaiEngine
{
	class NonMovable
	{
	public:
		NonMovable(NonMovable&&) = delete;
		NonMovable& operator=(NonMovable&&) = delete;

	protected:
		NonMovable() = default;
	};
}