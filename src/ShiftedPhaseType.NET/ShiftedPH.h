//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information. 
//
// Written by Luke Marshall and Timur Tankayev, 2019
//
#pragma once
#pragma unmanaged
#include "..\ShiftedPhaseType\ShiftedPhaseType.h"
#pragma managed
using namespace System;

//
// Provides a C# wrapper of a shifted phase type distribution.
// The original is written in C++ for performance benefits.
//
namespace ShiftedPhaseType {
namespace NET {
	public ref class ShiftedPH {

	//
	// Construction / Destruction
	//
	public:
		// Creates point mass at zero
		ShiftedPH() {
			shiftedPhaseType = new ShiftedPhaseTypeDistribution();
		}

		// Creates point mass at shift location
		ShiftedPH(double shift) {
			shiftedPhaseType = new ShiftedPhaseTypeDistribution(shift);
		}

		// Creates shifted phase type from 2 moments
		ShiftedPH(double m1, double m2, double shift) {
			shiftedPhaseType = new ShiftedPhaseTypeDistribution(m1, m2, shift);
		}

		// Copy constructor
		ShiftedPH(ShiftedPH^ other) {
			shiftedPhaseType = new ShiftedPhaseTypeDistribution(*(other->shiftedPhaseType));
		}

		//
		// Construct a phase type distribution from a mixture of erlangs.
		// Input is array of erlang distribution paramters and mixture probability (k, lambda, probability)
		//
		ShiftedPH(array<double>^ erlangs);


		// Finalizer and destructor
		!ShiftedPH()
		{
			delete shiftedPhaseType;
			shiftedPhaseType = NULL;
		}

		~ShiftedPH() {
			this->!ShiftedPH();
		}


	//
	// Add/Max operators
	//
	public:
		// Shift the phase type distribution
		ShiftedPH^ Add(double shift) {
			return gcnew ShiftedPH(shiftedPhaseType->Add(shift).release());
		}

		// Add two phase type distributions
		ShiftedPH^ Add(ShiftedPH^ other) {
			return Add(other, 0.0);
		}

		// Add two phase type distributions together with an extra shift
		ShiftedPH^ Add(ShiftedPH^ other, double shift) {
			return MaxWithAdd(other, shift, 0.0);
		}

		// Get distribution of the maximum between phase type and constant
		ShiftedPH^ MaxWith(double constant) {
			return gcnew ShiftedPH(shiftedPhaseType->MaxWith(constant).release());
		}

		// Returns max(this + other + shift, maxValue)
		ShiftedPH^ MaxWithAdd(ShiftedPH^ other, double shift, double maxValue) {
			return gcnew ShiftedPH(shiftedPhaseType->MaxWithAdd(*(other->shiftedPhaseType), shift, maxValue).release());
		}

		// Returns max(mult * this + shift, maxValue)
		ShiftedPH^ LinearMapWithMax(double mult, double shift, double maxValue) {
			return gcnew ShiftedPH(shiftedPhaseType->LinearMapWithMax(mult, shift, maxValue).release());
		}

	//
	// Member properties
	//
	public:
		// Returns the current shift value
		property double Shift {
			double get() {
				return shiftedPhaseType->Shift;
			}
		}

		// Returns the current Alpha
		property array<double>^ Alpha {
			array<double>^ get() {
				array<double>^ alpha = gcnew array<double>((int)shiftedPhaseType->Alpha->size());

				for (int i = 0; i < alpha->Length; ++i) {
					alpha[i] = shiftedPhaseType->Alpha->coeff(i);
				}

				return alpha;
			}
		}

		// Returns the current T Matrix
		property array<double,2>^ T {
			array<double,2>^ get() {
				array<double,2>^ T = gcnew array<double,2>((int)shiftedPhaseType->T->rows(), (int)shiftedPhaseType->T->cols());

				for (int i = 0; i < shiftedPhaseType->T->rows(); ++i) {
					for (int j = 0; j < shiftedPhaseType->T->cols(); ++j) {
						T[i,j] = shiftedPhaseType->T->coeff(i,j);
					}
				}

				return T;
			}
		}

		// Returns whether the distribution is a point mass
		property bool IsPointMass {
			bool get() {
				return shiftedPhaseType->IsPointMass();
			}
		}

		// Returns probability density at x
		double Density(double x) {
			return shiftedPhaseType->Density(x);
		}

		// Returns cumulative probability at x, i.e., P(X <= x)
		double CumulativeDistribution(double x) {
			return shiftedPhaseType->CumulativeDistribution(x);
		}

		// Returns the first moment of the 'unshifted' distribution
		property double Unshifted1Moment {
			double get() {
				return shiftedPhaseType->Unshifted1Moment();
			}
		}

		// Returns first and second moments of the 'unshifted' distribution
		property Tuple<double, double>^ Unshifted2Moments {
			Tuple<double, double>^ get() {
				auto m = shiftedPhaseType->Unshifted2Moments();
				return gcnew Tuple<double, double>(std::get<0>(m), std::get<1>(m));
			}
		}

		// Generates random sample of this shifted phase type
		double Sample() {
			return shiftedPhaseType->Sample();
		}

	//
	// Truncation operators
	//
	public:
		// Returns first and second moments of the left truncated distribution
		Tuple<double, double>^ LeftTruncMoments(double end) {
			auto m = shiftedPhaseType->LeftTruncMoments(end);
			return gcnew Tuple<double, double>(std::get<0>(m), std::get<1>(m));
		}

		// Returns first and second moments of the right truncated distribution
		Tuple<double, double>^ RightTruncMoments(double end) {
			auto m = shiftedPhaseType->RightTruncMoments(end);
			return gcnew Tuple<double, double>(std::get<0>(m), std::get<1>(m));
		}

		// Returns first moment of the right truncated distribution, multiplied by P(X >= end)
		double RightExpProb(double end) {
			return shiftedPhaseType->RightExpProb(end);
		}

		// Generates approximate shifted phase type from truncated distribution
		ShiftedPH^ ShiftedAndScaledPH(double end, double feasProb) {
			return gcnew ShiftedPH(shiftedPhaseType->ShiftedAndScaledPH(end, feasProb).release());
		}

		// Condition this distribution being greater than a constant
		ShiftedPH^ ConditionGreaterThan(double minValue) {
			return gcnew ShiftedPH(shiftedPhaseType->ConditionGreaterThan(minValue).release());
		}

	//
	// Private members
	//
	private:
		// The shifted phase type that we are wrapping
		ShiftedPhaseTypeDistribution* shiftedPhaseType;

		// Private constructor that takes a pointer to ShiftedPhaseTypeDistribution
		ShiftedPH(ShiftedPhaseTypeDistribution* take_ownership_ptr) {
			shiftedPhaseType = take_ownership_ptr;
		}
	};

}
}