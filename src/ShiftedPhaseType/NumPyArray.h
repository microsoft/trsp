//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information. 
//
// Written by Luke Marshall and Timur Tankayev, 2019
//
#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <iostream>

//
// Support loading data from numpy arrays (npy/npz files).
//
class NumPyArray
{
public:
	NumPyArray() : data_offset(0), header_offset(0), fortran_order(false) {
		raw_data.reset(new std::vector<char>());
	}

	// raw_data also includes the header information
	std::shared_ptr<std::vector<char>> raw_data;

	template<typename T> T* data() {
		return reinterpret_cast<T*>(&(*raw_data)[data_offset]);
	}

	std::string descr;
	bool fortran_order;
	std::vector<size_t> shape;

	size_t data_offset;
	size_t header_offset;

public:
	static std::map<std::string, NumPyArray> npz_load(std::string filename);
	static NumPyArray npy_load(std::string filename);

private:
	NumPyArray(std::vector<char>& buffer);
};

