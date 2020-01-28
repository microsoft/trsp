//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information. 
//
// Written by Luke Marshall and Timur Tankayev, 2019
//
#include "DistanceLookup.h"
#include <msclr\marshal_cppstd.h>

ShiftedPhaseType::NET::DistanceLookup::DistanceLookup(System::String^ file)
{
	// assumes that the matrix is in CSR format
	this->lookup = new ShiftedPhaseType::DistanceLookup(msclr::interop::marshal_as<std::string>(file));
}

