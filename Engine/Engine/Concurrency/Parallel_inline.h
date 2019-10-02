#pragma once

#include <iostream>
#include <future>
#include <vector>

namespace LaiEngine
{
	namespace Concurrency
	{
		template <typename IndexType, typename Function>
		static void ParallelFor(IndexType start, IndexType end, const Function& func)
		{
			if (start > end)
			{
				return;
			}

			// Estimate number of threads in the pool
			const static unsigned int numThreadsHint = std::thread::hardware_concurrency();
			const static unsigned int numThreads = numThreadsHint == 0u ? 8u : numThreadsHint;

			// Size of a slice for the range functions
			IndexType n = end - start + 1;
			IndexType slice = (IndexType)std::round(n / static_cast<double>(numThreads));
			slice = std::max(slice, IndexType(1));


			// Create pool and launch jobs
			std::vector<std::future<void>> pool;
			pool.reserve(numThreads);
			IndexType i1 = start;
			IndexType i2 = std::min(start + slice, end);


			// [Helper] Inner loop
			auto launchRange = [&func](size_t k1, size_t k2)
			{
				for (size_t k = k1; k < k2; k++)
				{
					func(k);
				}
			};


			for (size_t i = 0; i + 1 < numThreads && i1 < end; ++i)
			{
				pool.emplace_back(std::async(launchRange, i1, i2));
				i1 = i2;
				i2 = std::min(i2 + slice, end);
			}
			if (i1 < end)
			{
				pool.emplace_back(std::async(launchRange, i1, end));
			}


			//	 Wait for jobs to finish
			for (auto& f : pool)
			{
				if (f.valid())
				{
					f.wait();
				}
			}
		}

		// Serial version for easy comparison
		template<typename IndexType, typename Function>
		static void SequentialFor(IndexType start, IndexType end, const Function& func) 
		{
			for (IndexType i = start; i < end; i++)
			{
				func(i);
			}
		}
	}
}