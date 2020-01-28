//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information. 
//
// Written by Luke Marshall and Timur Tankayev, 2019
//
#pragma once

#include <algorithm>
#include <tuple>
#include <memory>
#include <vector>
#include <string>
#include "NumPyArray.h"

static const double INTERVAL_IN_MINUTES = 15.0;

namespace ShiftedPhaseType {

	//
	// Provides a wrapper of distance lookup routines, which includes loading a CSR sparse matrix (npz format).
	//
	class DistanceLookup {

	//
	// Construction / Destruction
	//
	public:
		DistanceLookup(std::string file);

	//
	// Travel time lookup operators
	//
	public:
		// Get the travel time distance from a to b at time t (measured in minutes)
		double GetDistance(int a, int b, double t)
		{
			int rowStart = rowIndexPtr[a * N + b];
			int rowEnd = rowIndexPtr[a * N + b + 1];

			// empty row returns 0
			if (rowStart == rowEnd)
				return 0;

			__int64 lb = lower_bound(rowStart, rowEnd, t);
			__int64 ub = upper_bound(rowStart, rowEnd, t);

			return linear_interpolate(lb, ub, t);
		}

		// Get all the travel time distances (and associated departure times) for travel from a to b
		std::tuple<std::vector<double>, std::vector<double>> GetProfile(int a, int b)
		{
			int rowStart = rowIndexPtr[a * N + b];
			int rowEnd = rowIndexPtr[a * N + b + 1];
			int size = rowEnd - rowStart;

			std::vector<double> time(size);
			std::vector<double> travel(size);

			for (int i = 0; i < size; ++i)
			{
				time[i] = colIndexPtr[rowStart + i];
				travel[i] = valuePtr[rowStart + i];
			}

			return std::make_tuple(time, travel);
		}

		// Fit a linear regression over the travel times between t1 and t2, when travelling from a to b
		std::tuple<double, double> FitDistance(int a, int b, double t1, double t2)
		{
			// handle single time point
			if (t1 == t2)
				return std::make_tuple<double, double>(GetDistance(a, b, t1), 0);

			// find segment and calc linear regression
			int rowStart = rowIndexPtr[a * N + b];
			int rowEnd = rowIndexPtr[a * N + b + 1];

			// empty row returns 0
			if (rowStart == rowEnd)
				return std::make_tuple<double, double>(0, 0);

			__int64 lb1 = lower_bound(rowStart, rowEnd, t1);
			__int64 ub1 = upper_bound(rowStart, rowEnd, t1);

			__int64 lb2 = lower_bound(rowStart, rowEnd, t2);
			__int64 ub2 = upper_bound(rowStart, rowEnd, t2);

			// if t1, t2 are outside our traffic matrix data, then return the maximum travel time
			double maxTravel = (lb1 < ub1 || lb2 < ub2) ? 0 : *std::max_element(valuePtr + rowStart, valuePtr + rowEnd);

			double y1 = lb1 < ub1 ? linear_interpolate(lb1, ub1, t1) : maxTravel;
			double y2 = lb2 < ub2 ? linear_interpolate(lb2, ub2, t2) : maxTravel;

			double xmean = t2 - t1, ymean = y1 + y2;
			int count = 2;

			for (__int64 i = ub1; i < lb2 || i == lb2 && time_value(lb2) < t2; ++i)
			{
				xmean += time_value(i) - t1;
				ymean += valuePtr[i];
				++count;
			}

			xmean /= count;
			ymean /= count;

			double m1 = (0 - xmean) * (y1 - ymean) + (t2 - t1 - xmean) * (y2 - ymean);
			double m2 = (0 - xmean) * (0  - xmean) + (t2 - t1 - xmean) * (t2 - t1 - xmean);

			for (__int64 i = ub1; i < lb2 || i == lb2 && time_value(lb2) < t2; ++i)
			{
				m1 += (time_value(i) - t1 - xmean) * (valuePtr[i] - ymean);
				m2 += (time_value(i) - t1 - xmean) * (time_value(i) - t1 - xmean);
			}

			double slope = m1 / m2;
			double shift = ymean - slope * xmean;

			return std::make_tuple(shift, slope);
		}

	//
	// Private functions
	//
	private:
		// Linearly interpolate t between colIndexPtr[lb] and colIndexPtr[ub]
		inline double linear_interpolate(__int64 lb, __int64 ub, double t)
		{
			return valuePtr[lb] + (static_cast<double>(valuePtr[ub]) - static_cast<double>(valuePtr[lb])) * (t - time_value(lb)) / (time_value(ub) - time_value(lb));
		}

		// Find the index 'lb' such that colIndexPtr[lb] <= t
		inline __int64 lower_bound(int rowStart, int rowEnd, double t)
		{
			__int64 lb = std::lower_bound(colIndexPtr + rowStart, colIndexPtr + rowEnd, t / INTERVAL_IN_MINUTES) - colIndexPtr;
			return lb - int(time_value(lb) > t);  // subtract one if t is not exactly on the element (true lower bound)
		}

		// Find the index 'ub' such that colIndexPtr[ub] > t
		inline __int64 upper_bound(int rowStart, int rowEnd, double t)
		{
			return std::upper_bound(colIndexPtr + rowStart, colIndexPtr + rowEnd, t / INTERVAL_IN_MINUTES) - colIndexPtr;
		}

		// Return the actual time at the given index (appropriately cast to a double)
		inline double time_value(__int64 index)
		{
			return static_cast<__int64>(colIndexPtr[index]) * INTERVAL_IN_MINUTES;
		}

	//
	// Private members
	//
	private:
		int N;

		NumPyArray values;
		NumPyArray rowIndices;
		NumPyArray colIndices;

		int* rowIndexPtr;
		int* colIndexPtr;
		float* valuePtr;
	};
};