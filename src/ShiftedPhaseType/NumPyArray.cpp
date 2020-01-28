#include "NumPyArray.h"

#include <fstream>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/tokenizer.hpp>
#include <sys/stat.h>

// Decompress a memory buffer to a pre-allocated memory buffer
void inflate(const char* src, char* dst, uint32_t compressed_size, uint32_t uncompressed_size)
{
	boost::iostreams::zlib_decompressor(-15).filter()
		.filter(src, src + compressed_size, dst, dst + uncompressed_size, false);
}

// Load numpy arrays from a zip archive
std::map<std::string, NumPyArray> NumPyArray::npz_load(std::string filename)
{
	std::map<std::string, NumPyArray> all;
	std::ifstream input_file(filename, std::ios_base::in | std::ios_base::binary);

	// read ZIP file structure: https://en.wikipedia.org/wiki/Zip_(file_format)
	std::vector<char> local_header(30);

	while (true) {
		input_file.read(&local_header[0], 30);

		if (input_file.gcount() != 30) {
			throw std::runtime_error("npz_load: failed read");
		}

		// stop if we've reached the end (local header magic number)
		if (*reinterpret_cast<uint32_t*>(&local_header[0]) != 0x04034b50) {
			break;
		}

		// get npy variable name
		std::string variable_name;
		variable_name.resize(*reinterpret_cast<uint16_t*>(&local_header[26]));

		input_file.read(&variable_name[0], variable_name.size());

		if (input_file.gcount() != variable_name.size()) {
			throw std::runtime_error("npz_load: failed read");
		}

		variable_name.erase(variable_name.end() - 4, variable_name.end()); // remove file extension (.npy)

		// skip the extra field
		input_file.seekg(*reinterpret_cast<uint16_t*>(&local_header[28]), input_file.cur);

		// load data (uncompress if needed)
		uint16_t compression_method = *reinterpret_cast<uint16_t*>(&local_header[8]);
		uint32_t compressed_bytes = *reinterpret_cast<uint32_t*>(&local_header[18]);
		uint32_t uncompressed_bytes = *reinterpret_cast<uint32_t*>(&local_header[22]);

		std::vector<char> compressed(compressed_bytes);
		input_file.read(compressed.data(), compressed_bytes);

		if (compression_method == 0)
		{
			all[variable_name] = NumPyArray(compressed);
		}
		else
		{
			std::vector<char> uncompressed(uncompressed_bytes);
			inflate(compressed.data(), uncompressed.data(), compressed_bytes, uncompressed_bytes);

			all[variable_name] = NumPyArray(uncompressed);
		}
	}

	return all;
}

// Loads a single numpy array from file
NumPyArray NumPyArray::npy_load(std::string filename)
{
	struct stat st;
	stat(filename.c_str(), &st);

	std::ifstream input_file(filename, std::ios_base::in | std::ios_base::binary);

	std::vector<char> raw_data(st.st_size);
	input_file.read(raw_data.data(), st.st_size);

	return NumPyArray(raw_data);
}

// Constructs a numpy array from a memory buffer (takes ownership of the memory buffer for performance)
// https://github.com/numpy/numpy/blob/067cb067cb17a20422e51da908920a4fbb3ab851/doc/neps/nep-0001-npy-format.rst
// https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
NumPyArray::NumPyArray(std::vector<char>& buffer) : data_offset(0), header_offset(0), fortran_order(false)
{
	// check to see if buffer contains a valid numpy array
	if (strncmp("\x93NUMPY", buffer.data(), 6) != 0)
	{
		throw std::runtime_error("File is not a numpy array");
	}

	size_t header_size = 0, header_size_bytes = 0;

	// header length: 2 bytes in v1, 4 bytes in v2 and v3
	if (buffer[6] == 1)
	{
		header_size_bytes = 2;
		header_size = *reinterpret_cast<int16_t*>(buffer.data() + 8);
	}
	else if (buffer[6] == 2 || buffer[6] == 3)
	{
		header_size_bytes = 4;
		header_size = *reinterpret_cast<int32_t*>(buffer.data() + 8);
	}
	else
	{
		throw std::runtime_error("Version not supported");
	}

	this->header_offset = 8 + header_size_bytes;
	this->data_offset = this->header_offset + header_size;

	// parse array header
	std::string header(buffer.data() + this->header_offset, header_size);

	std::replace(header.begin() + header.find('('), header.begin() + header.find(')'), ',', ';');
	boost::tokenizer<boost::char_separator<char>> tokens(header, boost::char_separator<char>("{,: }"));

	for (auto it = tokens.begin(); it != tokens.end(); ++it)
	{
		// descr: dtype
		if (it->compare("'descr'") == 0) {
			++it;
			this->descr = it->substr(1, it->length() - 2);
		}

		// fortran_order : bool
		if (it->compare("'fortran_order'") == 0) {
			++it;
			this->fortran_order = it->compare("True") == 0;
		}

		// shape: int tuple
		if (it->compare("'shape'") == 0) {
			++it;
			
			for (const auto& s : boost::tokenizer<boost::char_separator<char>>(*it, boost::char_separator<char>("(;)"))) {
				this->shape.push_back(atoi(s.c_str()));
			}
		}
	}

	// take ownership of buffer memory
	this->raw_data.reset(new std::vector<char>());
	this->raw_data->swap(buffer);
}
