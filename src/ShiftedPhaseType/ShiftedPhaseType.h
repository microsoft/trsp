//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information. 
//
// Written by Luke Marshall and Timur Tankayev, 2019
//
#pragma once

#include "SparseLoad.h"
#include <random>
#include <memory>
#include <tuple>
#include <vector>

// Precision for vector * matrix exponential calculations
#define PREC 1e-6;

namespace ShiftedPhaseType {

	//
	// Implements a shifted phase type distribution
	//
	class ShiftedPhaseTypeDistribution
	{
		//
		// Member variables
		//
	public:
		double Shift, GreaterThanConditionalProbability, GreaterThanConditionalScale;
		std::unique_ptr<RowVectorXd> Alpha;
		std::unique_ptr<SpMat> T;

		//
		// Constructors
		//
	public:
		// Creates point mass at shift location
		ShiftedPhaseTypeDistribution(double shift = 0.0);

		// Creates shifted phase type from 2 moments
		ShiftedPhaseTypeDistribution(double m1, double m2, double shift);

		// Creates shifted phase type from given parameters
		ShiftedPhaseTypeDistribution(double shift, RowVectorXd& alpha, SpMat& t, double greaterThanConditionalProbability = 0.0, double greaterThanConditionalScale = 1.0);

		// Copy constructor
		ShiftedPhaseTypeDistribution(const ShiftedPhaseTypeDistribution& other);

		//
		// Add/Max operators
		//
	public:
		// Shift the phase type distribution
		inline std::unique_ptr<ShiftedPhaseTypeDistribution> Add(double other) {
			return std::make_unique<ShiftedPhaseTypeDistribution>(Shift + other, *Alpha, *T);
		}

		// Add two phase type distributions together with an extra shift
		inline std::unique_ptr<ShiftedPhaseTypeDistribution> Add(ShiftedPhaseTypeDistribution& other, double shift = 0.0) {
			return MaxWithAdd(other, shift, 0.0);
		}

		// Get distribution of the maximum between phase type and constant
		std::unique_ptr<ShiftedPhaseTypeDistribution> MaxWith(double constant);

		// Returns max(this + other + shift, maxValue)
		std::unique_ptr<ShiftedPhaseTypeDistribution> MaxWithAdd(ShiftedPhaseTypeDistribution& other, double shift, double maxValue);

		// Returns max(mult * this + shift, maxValue)
		std::unique_ptr<ShiftedPhaseTypeDistribution> LinearMapWithMax(double mult, double shift, double maxValue);

		//
		// Member properties
		//
	public:
		// Returns whether the distribution is a point mass
		bool IsPointMass() { return Alpha->sum() < 1e-8; }

		// Returns probability density at x
		double Density(double x);

		// Returns cumulative probability at x, i.e., P(X <= x)
		double CumulativeDistribution(double x);

		// Returns the first moment of the 'unshifted' distribution
		double Unshifted1Moment();

		// Returns first and second moments of the 'unshifted' distribution
		std::tuple<double, double> Unshifted2Moments();

		// Generates random sample of this shifted phase type
		double Sample();

		//
		// Truncation operators
		//
	public:
		// Returns first and second moments of the left truncated distribution
		std::tuple<double, double> LeftTruncMoments(double end);

		// Returns first and second moments of the right truncated distribution
		std::tuple<double, double> RightTruncMoments(double end);

		// Returns first moment of the right truncated distribution, multiplied by P(X >= end)
		double RightExpProb(double end);

		// Generates approximate shifted phase type from truncated distribution
		std::unique_ptr<ShiftedPhaseTypeDistribution> ShiftedAndScaledPH(double end, double feasProb);

		// Condition this distribution being greater than a constant
		std::unique_ptr<ShiftedPhaseTypeDistribution> ConditionGreaterThan(double minValue);

		//
		// Private members
		//
	private:
		// helper functions for vector * matrix exponential routines
		double h(double x);
		double hifunc(double rho, double B, double loge);
		double lofunc(double rho, double A, double loge);

		int get_mlo(int mhi, double rho);
		int get_m(double rho, double prec);
		RowVectorXd vexpQ(RowVectorXd& v, SpMat& Q, double scaleQ);

		// fast row sum calculation for sparse matrix in column compressed format
		std::vector<double> row_sum(SpMat* T);

		// random number generator for shifted phase type sample generation
		std::default_random_engine* generator = NULL;
	};
};
