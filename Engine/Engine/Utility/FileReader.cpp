#include "FileReader.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <sstream>
#include <stdexcept>

namespace LaiEngine
{
	namespace Utility
	{
		std::string FileReader(const std::string & path)
		{
			std::ifstream inFile(path);

			if (!inFile.is_open())
			{
				throw std::runtime_error("Unable to open file: " + path);
			}

			std::stringstream stream;

			stream << inFile.rdbuf();

			std::string res = stream.str();

			stream.clear();

			return res;
		}
	}
}