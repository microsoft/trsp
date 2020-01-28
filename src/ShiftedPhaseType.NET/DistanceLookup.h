//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information. 
//
// Written by Luke Marshall and Timur Tankayev, 2019
//
#pragma once

#pragma unmanaged
#include "..\ShiftedPhaseType\DistanceLookup.h"
#include <memory>
#include <vector>
#include <tuple>
#pragma managed

//
// Provides a C# wrapper of distance lookup routines, which includes loading a CSR sparse matrix (npz format).
//
namespace ShiftedPhaseType {
namespace NET {

	public ref class DistanceLookup {

	//
	// Construction / Destruction
	//
	public:
		DistanceLookup(System::String^ file);

		// Finalizer and destructor
		!DistanceLookup()
		{
			delete lookup;
			lookup = NULL;
		}

		~DistanceLookup() {
			this->!DistanceLookup();
		}

	//
	// Travel time lookup operators
	//
	public:
		// Get the travel time distance from a to b at time t (measured in minutes)
		double GetDistance(int a, int b, double t)
		{
			return lookup->GetDistance(a, b, t);
		}

		// Get all the travel time distances (and associated departure times) for travel from a to b
		System::Tuple<array<double>^, array<double>^>^ GetProfile(int a, int b)
		{
			std::tuple<std::vector<double>, std::vector<double>> result = lookup->GetProfile(a, b);
			int size = (int)std::get<0>(result).size();

			array<double>^ time = gcnew array<double>(size);
			array<double>^ travel = gcnew array<double>(size);

			for (int i = 0; i < size; ++i)
			{
				time[i] = std::get<0>(result)[i];
				travel[i] = std::get<1>(result)[i];
			}

			return gcnew System::Tuple<array<double>^, array<double>^>(time, travel);
		}

		// Fit a linear regression over the travel times between t1 and t2, when travelling from a to b
		System::Tuple<double, double>^ FitDistance(int a, int b, double t1, double t2)
		{
			std::tuple<double, double> result = lookup->FitDistance(a, b, t1, t2);
			return gcnew System::Tuple<double, double>(std::get<0>(result), std::get<1>(result));
		}

	//
	// Private members
	//
	private:
		ShiftedPhaseType::DistanceLookup* lookup;
	};
}
}