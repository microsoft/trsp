//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information. 
//
// Written by Luke Marshall and Timur Tankayev, 2019
//
#include "DistanceLookup.h"
using namespace ShiftedPhaseType;

DistanceLookup::DistanceLookup(std::string file)
{
	// assumes that the matrix is in CSR format
	std::map<std::string, NumPyArray> arr = NumPyArray::npz_load(file);
	
	__int32* size = arr["shape"].data<__int32>();
	this->N = int(sqrt(size[0]));
	
	// 'takes ownership' of the memory, rather than copy
	this->values = arr["data"];
	this->colIndices = arr["indices"];
	this->rowIndices = arr["indptr"];
	
	// provides nice access to memory block
	this->valuePtr = this->values.data<float>();
	this->colIndexPtr = this->colIndices.data<__int32>();
	this->rowIndexPtr = this->rowIndices.data<__int32>();
}

