//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information. 
//
// Written by Luke Marshall and Timur Tankayev, 2019
//
#pragma unmanaged
#include "ShiftedPhaseType.h"
#include <chrono>
#include <cmath>
#include <boost/math/special_functions/gamma.hpp>

using namespace ShiftedPhaseType;

//
// Constructors
//

ShiftedPhaseTypeDistribution::ShiftedPhaseTypeDistribution(double shift)
{
	this->Shift = shift;
	this->GreaterThanConditionalProbability = 0.0;
	this->GreaterThanConditionalScale = 1.0;

	this->Alpha = std::make_unique<RowVectorXd>(1);
	this->Alpha->fill(0);

	this->T = std::make_unique<SpMat>(1, 1);
	this->T->setIdentity();
}

ShiftedPhaseTypeDistribution::ShiftedPhaseTypeDistribution(double shift, RowVectorXd &a, SpMat &t, double greaterThanConditionalProbability, double greaterThanConditionalScale)
{
	this->Shift = shift;
	this->GreaterThanConditionalProbability = greaterThanConditionalProbability;
	this->GreaterThanConditionalScale = greaterThanConditionalScale;
	this->Alpha = std::make_unique<RowVectorXd>(a);
	this->T = std::make_unique<SpMat>(t);
}

//
// Horváth, Gábor, and Miklós Telek. "BuTools 2: a Rich Toolbox for Markovian Performance Evaluation." In VALUETOOLS. 2016.
//
ShiftedPhaseTypeDistribution::ShiftedPhaseTypeDistribution(double m1, double m2, double shift)
{
	double cv2 = m2 / m1 / m1 - 1.0;
	double lambd = 1.0 / m1;

	size_t N = (int)fmax(ceil(1.0 / cv2), 2.0);
	double p = 1.0 / (cv2 + 1.0 + (cv2 - 1.0) / (N - 1));
	double lambda = lambd * p * N;

	// load sparse matrix in CCO format
	this->T = std::make_unique<SpMat>(N, N);
	SparseLoad loader(this->T.get(), 2 * N - 1);

	for (int i = 0; i < N - 1; ++i)
	{
		if (i > 0)
		{
			loader.add(lambda, i - 1);
		}

		loader.add(-lambda, i);
		loader.next_col();
	}

	if (N > 1)
	{
		loader.add(lambda, N - 2);
	}

	loader.add(-lambd * N, N - 1);
	loader.next_col();

	// load alpha
	this->Alpha = std::make_unique<RowVectorXd>(N);
	this->Alpha->fill(0.0);

	(*this->Alpha)(0) = p;
	(*this->Alpha)(N - 1) = 1.0 - p;

	// load shift
	this->Shift = shift;
	this->GreaterThanConditionalProbability = 0.0;
	this->GreaterThanConditionalScale = 1.0;
}

ShiftedPhaseTypeDistribution::ShiftedPhaseTypeDistribution(const ShiftedPhaseTypeDistribution& other)
{
	this->Shift = other.Shift;
	this->GreaterThanConditionalProbability = other.GreaterThanConditionalProbability;
	this->GreaterThanConditionalScale = other.GreaterThanConditionalScale;

	this->Alpha = std::make_unique<RowVectorXd>(*other.Alpha);
	this->T = std::make_unique<SpMat>(*other.T);
}


//
// Add/Max operators
//

std::unique_ptr<ShiftedPhaseTypeDistribution> ShiftedPhaseTypeDistribution::MaxWith(double constant)
{
	if (Shift < constant)
	{
		// Deal with point-mass 'like' phase types
		if (this->IsPointMass()) {
			return std::make_unique<ShiftedPhaseTypeDistribution>(constant, *this->Alpha, *this->T, this->GreaterThanConditionalProbability, this->GreaterThanConditionalScale);
		}
		else {
			RowVectorXd tmp = vexpQ(*this->Alpha, *this->T, constant - Shift);
			return std::make_unique<ShiftedPhaseTypeDistribution>(constant, tmp, *this->T, this->GreaterThanConditionalProbability, this->GreaterThanConditionalScale);
		}
	}
	else
	{
		return std::make_unique<ShiftedPhaseTypeDistribution>(Shift, *this->Alpha, *this->T, this->GreaterThanConditionalProbability, this->GreaterThanConditionalScale);
	}
}

std::unique_ptr<ShiftedPhaseTypeDistribution> ShiftedPhaseTypeDistribution::MaxWithAdd(ShiftedPhaseTypeDistribution& other, double shift, double maxValue)
{
	double newShift = this->Shift + other.Shift + shift;

	// Check if either conditional distribution is infeasible
	if (this->GreaterThanConditionalProbability >= 1.0 || other.GreaterThanConditionalProbability >= 1.0)
	{
		RowVectorXd newA(1);
		newA.fill(0);

		SpMat newT(1, 1);
		newT.setIdentity();

		return std::make_unique<ShiftedPhaseTypeDistribution>(maxValue, newA, newT, 1.0);
	}

	// Deal with point-mass 'like' phase types
	if (this->IsPointMass())
	{
		if (newShift < maxValue)
		{
			if (other.IsPointMass()) {
				return std::make_unique<ShiftedPhaseTypeDistribution>(maxValue, *(other.Alpha), *(other.T), std::max(this->GreaterThanConditionalProbability, other.GreaterThanConditionalProbability));
			}
			else {
				RowVectorXd tmp = vexpQ(*(other.Alpha), *(other.T), maxValue - newShift);
				return std::make_unique<ShiftedPhaseTypeDistribution>(maxValue, tmp, *(other.T), std::max(this->GreaterThanConditionalProbability, other.GreaterThanConditionalProbability), other.GreaterThanConditionalScale);
			}
		}
		else
		{
			return std::make_unique<ShiftedPhaseTypeDistribution>(newShift, *(other.Alpha), *(other.T), std::max(this->GreaterThanConditionalProbability, other.GreaterThanConditionalProbability), other.GreaterThanConditionalScale);
		}
	}
	else if (other.IsPointMass())
	{
		if (newShift < maxValue) {
			RowVectorXd tmp = vexpQ(*this->Alpha, *this->T, maxValue - newShift);
			return std::make_unique<ShiftedPhaseTypeDistribution>(maxValue, tmp, *this->T, std::max(this->GreaterThanConditionalProbability, other.GreaterThanConditionalProbability), this->GreaterThanConditionalScale);
		}
		else {
			return std::make_unique<ShiftedPhaseTypeDistribution>(newShift, *this->Alpha, *this->T, std::max(this->GreaterThanConditionalProbability, other.GreaterThanConditionalProbability), this->GreaterThanConditionalScale);
		}
	}
	else
	{
		RowVectorXd newA(this->Alpha->size() + other.Alpha->size());
		newA << *this->Alpha, (1 - this->Alpha->sum() - this->GreaterThanConditionalProbability) * (*(other.Alpha));

		int thisRows = (int)this->T->rows();
		int thisCols = (int)this->T->cols();

		SpMat newT(thisRows + other.T->rows(), thisCols + other.T->cols());

		// (-T) * ones(T->cols) * other->A;
		std::vector<double> sumT = row_sum(this->T.get());
		std::vector<int> indexT(this->T->cols());

		// gather non zeros for T row sum
		int nnzT = 0;
		for (int i = 0; i < this->T->cols(); ++i)
		{
			if (sumT[i] != 0)  // tolerance?
			{
				indexT[nnzT] = i;
				++nnzT;
			}
		}

		SparseLoad loader(&newT, (int)(this->T->nonZeros() + other.T->nonZeros() + nnzT * other.Alpha->nonZeros()));

		// load first block
		loader.initial_block(this->T.get());
		loader.next_col();

		// load upper-right and bottom blocks
		double* lowerValue = other.T->valuePtr();
		int* lowerIndex = other.T->innerIndexPtr();
		int* lowerOuter = other.T->outerIndexPtr() + 1;

		for (int j = 0; j < other.T->cols(); ++j)
		{
			// add upper column
			double aj = (*(other.Alpha))(j);

			if (aj != 0)
			{
				for (int i = 0; i < nnzT; ++i)
				{
					loader.add((-sumT[indexT[i]])*aj, indexT[i]);
				}
			}

			// add lower column
			for (int i = *lowerOuter - *(lowerOuter - 1); i > 0; --i)
			{
				loader.add(*lowerValue, *lowerIndex + (int)T->rows());
				++lowerValue;
				++lowerIndex;
			}
			++lowerOuter;
			loader.next_col();
		}
		
		// maxWith
		if (newShift < maxValue) {
			RowVectorXd tmp = vexpQ(newA, newT, maxValue - newShift);
			return std::make_unique<ShiftedPhaseTypeDistribution>(maxValue, tmp, newT, this->GreaterThanConditionalProbability + other.GreaterThanConditionalProbability, this->GreaterThanConditionalScale * other.GreaterThanConditionalScale);
		}
		else {
			return std::make_unique<ShiftedPhaseTypeDistribution>(newShift, newA, newT, this->GreaterThanConditionalProbability + other.GreaterThanConditionalProbability, this->GreaterThanConditionalScale * other.GreaterThanConditionalScale);
		}
	}
}

std::unique_ptr<ShiftedPhaseTypeDistribution> ShiftedPhaseTypeDistribution::LinearMapWithMax(double mult, double shift, double maxValue)
{
	static const double POINT_MASS_THRESHOLD = 0.1;
	double newShift = Shift + shift;

	if (mult < 0)
	{
		throw std::runtime_error("LinearMapWithMax must have positive multiplicative coefficient.");
	}

	if (this->IsPointMass())
	{
		return std::make_unique<ShiftedPhaseTypeDistribution>(newShift < maxValue ? maxValue : newShift, *this->Alpha, *this->T, this->GreaterThanConditionalProbability, this->GreaterThanConditionalScale);
	}
	else if (mult < POINT_MASS_THRESHOLD)
	{
		// Multiplicative coefficent is so small that it essentially reduces the variance to zero.
		RowVectorXd newA(1);
		newA.fill(0);

		SpMat newT(1, 1);
		newT.setIdentity();

		return std::make_unique<ShiftedPhaseTypeDistribution>(newShift < maxValue ? maxValue : newShift, newA, newT, this->GreaterThanConditionalProbability, this->GreaterThanConditionalScale);
	}
	else
	{
		SpMat newT(*T / mult);

		if (newShift < maxValue) {
			RowVectorXd tmp = vexpQ(*this->Alpha, newT, maxValue - newShift);
			return std::make_unique<ShiftedPhaseTypeDistribution>(maxValue, tmp, newT, this->GreaterThanConditionalProbability, this->GreaterThanConditionalScale);
		}
		else {
			return std::make_unique<ShiftedPhaseTypeDistribution>(newShift, *this->Alpha, newT, this->GreaterThanConditionalProbability, this->GreaterThanConditionalScale);
		}
	}
}


//
// Member properties
//

double ShiftedPhaseTypeDistribution::Density(double x)
{
	if (x == this->Shift && this->IsPointMass())
	{
		return 1.0 - this->GreaterThanConditionalProbability;
	}
	else if (x <= this->Shift)
	{
		return 0.0;
	}
	else
	{
		RowVectorXd s = vexpQ(*this->Alpha, *this->T, x - this->Shift);
		return this->GreaterThanConditionalScale * s * (-*this->T) * VectorXd::Ones(this->T->rows());
	}
}

double ShiftedPhaseTypeDistribution::CumulativeDistribution(double x)
{
	if (x < this->Shift)
	{
		return 0;
	}
	else if (this->IsPointMass() || this->Shift == x)
	{
		return 1.0 - this->Alpha->sum() - this->GreaterThanConditionalProbability;
	}
	else
	{
		auto s = vexpQ(*this->Alpha, *this->T, x - this->Shift);
		return this->GreaterThanConditionalScale * (1.0 - s.dot(VectorXd::Ones(T->rows())) - this->GreaterThanConditionalProbability);
	}
}

double ShiftedPhaseTypeDistribution::Unshifted1Moment()
{
	if (this->IsPointMass())
		return 0;

	return -Alpha->dot(this->T->triangularView<Eigen::Upper>().solve(VectorXd::Ones(this->T->rows())));
}

std::tuple<double, double> ShiftedPhaseTypeDistribution::Unshifted2Moments()
{
	if (this->IsPointMass())
		return std::make_tuple(0, 0);

	VectorXd TsolveOne = this->T->triangularView<Eigen::Upper>().solve(VectorXd::Ones(this->T->rows()));
	double m1 = -Alpha->dot(TsolveOne);
	double m2 = 2 * Alpha->dot(this->T->triangularView<Eigen::Upper>().solve(TsolveOne));

	return std::make_tuple(m1, m2);
}

double ShiftedPhaseTypeDistribution::Sample()
{
	// ensure random number generator is available
	if (generator == NULL)
	{
		generator = new std::default_random_engine((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
	}

	if (this->IsPointMass())
		return Shift;

	double time = 0.0;
	int state = 0;

	std::uniform_real_distribution<double> u(0.0, 1.0);

	double initSeed = u(*generator);
	double cumSum = 0.0;

	for (int i = 0; i < this->Alpha->size(); ++i)
	{
		cumSum += (*this->Alpha)(i);

		if (cumSum > initSeed)
		{
			break;
		}

		++state;
	}

	while (state < this->Alpha->size())
	{
		time += log(u(*generator)) / T->coeff(state, state);

		double nextSeed = u(*generator) * (-T->coeff(state, state));
		double nextCumSum = 0.0;
		int nextState = state;

		for (int j = state + 1; j < this->Alpha->size(); j++)
		{
			nextCumSum += T->coeff(state, j);

			if (nextCumSum > nextSeed)
			{
				nextState = j;
				break;
			}
		}

		if (state == nextState)
		{
			state = (int)this->Alpha->size();
		}
		else
		{
			state = nextState;
		}
	}

	return time + Shift;
}

//
// Truncation operators
//

std::tuple<double, double> ShiftedPhaseTypeDistribution::LeftTruncMoments(double end)
{
	double x = end - Shift;

	VectorXd one = VectorXd::Ones(T->rows());
	RowVectorXd AexpxT = vexpQ(*this->Alpha, *T, x);

	VectorXd TinvOne = T->triangularView<Eigen::Upper>().solve(one);
	VectorXd Tinv2One = T->triangularView<Eigen::Upper>().solve(TinvOne);

	double m1 = -x * AexpxT.dot(one) + AexpxT.dot(TinvOne) - this->Alpha->dot(TinvOne);
	double m2 = -x * x * AexpxT.dot(one) + 2 * x * AexpxT.dot(TinvOne) - 2 * AexpxT.dot(Tinv2One) + 2 * this->Alpha->dot(Tinv2One);

	return std::make_tuple(m1, m2);
}

std::tuple<double, double> ShiftedPhaseTypeDistribution::RightTruncMoments(double end)
{
	double x = end - Shift;

	if (x <= 0)
	{
		return Unshifted2Moments();
	}

	VectorXd one = VectorXd::Ones(T->rows());
	RowVectorXd AexpxT = vexpQ(*this->Alpha, *this->T, x);

	VectorXd TinvOne = T->triangularView<Eigen::Upper>().solve(one);
	VectorXd Tinv2One = T->triangularView<Eigen::Upper>().solve(TinvOne);

	double m1 = -AexpxT.dot(TinvOne) + x * AexpxT.dot(one);
	double m2 = 2 * AexpxT.dot(Tinv2One) - 2 * x * AexpxT.dot(TinvOne) + x * x * AexpxT.dot(one);

	return std::make_tuple(m1, m2);
}

double ShiftedPhaseTypeDistribution::RightExpProb(double end)
{
	if (this->IsPointMass())
	{
		return Shift > end ? Shift - end : 0;
	}

	if (Shift >= end)
	{
		return Shift + Unshifted1Moment() - end;
	}

	double x = end - Shift;

	VectorXd one = VectorXd::Ones(this->T->rows());
	RowVectorXd AexpxT = vexpQ(*this->Alpha, *this->T, x);
	VectorXd TinvOne = this->T->triangularView<Eigen::Upper>().solve(one);

	return -AexpxT.dot(TinvOne);
}

std::unique_ptr<ShiftedPhaseTypeDistribution> ShiftedPhaseTypeDistribution::ShiftedAndScaledPH(double end, double feasProb)
{
	// Do nothing for point mass
	if (this->IsPointMass())
		return std::make_unique<ShiftedPhaseTypeDistribution>(*this);

	// Speed up calculations by sharing partial results
	VectorXd one = VectorXd::Ones(T->rows());
	VectorXd TinvOne = this->T->triangularView<Eigen::Upper>().solve(one);
	VectorXd Tinv2One = this->T->triangularView<Eigen::Upper>().solve(TinvOne);

	double x = end - Shift;
	RowVectorXd AexpxT = vexpQ(*this->Alpha, *T, x);

	double m1 = -Alpha->dot(TinvOne);
	double m2 = 2 * Alpha->dot(Tinv2One);

	double xAexpxTOne = x * AexpxT.dot(one);
	double AexpxTTinvOne = AexpxT.dot(TinvOne);

	double mh1 = m1 - xAexpxTOne + AexpxTTinvOne;
	double mh2 = m2 - x * xAexpxTOne + 2 * x * AexpxTTinvOne - 2 * AexpxT.dot(Tinv2One);

	mh1 /= feasProb;
	mh2 /= feasProb;

	auto theta = sqrt((m2 - m1 * m1) / (mh2 - mh1 * mh1));
	auto gamma = mh1 - m1 / theta;

	SpMat newT = theta * (*this->T);
	return std::make_unique<ShiftedPhaseTypeDistribution>(this->Shift + gamma, *this->Alpha, newT);
}

std::unique_ptr<ShiftedPhaseTypeDistribution> ShiftedPhaseTypeDistribution::ConditionGreaterThan(double minValue)
{
	const double precision = PREC;

	if (Shift < minValue)
	{
		if (this->IsPointMass()) {
			// if point mass is less than minValue, then the GreaterThan conditional probability is zero
			return std::make_unique<ShiftedPhaseTypeDistribution>(minValue, *this->Alpha, *this->T, 1.0);
		}
		else {
			RowVectorXd tmp = vexpQ(*this->Alpha, *this->T, minValue - Shift);
			double newGreaterThanConditionalProbability = 1.0 - tmp.sum();

			if (newGreaterThanConditionalProbability >= 1.0 - precision)
			{
				// Conditional distribution is essentially nothing, so change to a zero point mass
				RowVectorXd newAlpha(1);
				newAlpha.fill(0);

				SpMat newT(1,1);
				newT.setIdentity();

				return std::make_unique<ShiftedPhaseTypeDistribution>(minValue, newAlpha, newT, 1.0);
			}
			else
			{
				double p2 = this->GreaterThanConditionalScale * (newGreaterThanConditionalProbability - this->GreaterThanConditionalProbability);
				double newGreaterThanConditionalScale = this->GreaterThanConditionalScale / (1.0 - p2);

				return std::make_unique<ShiftedPhaseTypeDistribution>(minValue, tmp, *T, newGreaterThanConditionalProbability, newGreaterThanConditionalScale);
			}
		}
	}
	else
	{
		return std::make_unique<ShiftedPhaseTypeDistribution>(Shift, *this->Alpha.get(), *this->T, this->GreaterThanConditionalProbability, this->GreaterThanConditionalScale);
	}
}


//
// Private members
//

inline double ShiftedPhaseTypeDistribution::h(double x)
{
	return 1.0 - x + x * log(x);
}

inline double ShiftedPhaseTypeDistribution::hifunc(double rho, double B, double loge)
{
	return rho + (B - loge) / 3.0*(1.0 + sqrt(1.0 + 18.0*rho / (B - loge)));
}

inline double ShiftedPhaseTypeDistribution::lofunc(double rho, double A, double loge)
{
	const double logroot2pi = 0.5*log(2 * 3.141592653589793);
	return rho + sqrt(2 * rho)*sqrt(-logroot2pi - loge - 1.5*log(A) + log(A - 1));
}

int ShiftedPhaseTypeDistribution::get_mlo(int mhi, double rho)
{
	int mlo = 2 * (int)(rho - 0.5) - mhi;
	return mlo < 0 ? 0 : mlo;
}

int ShiftedPhaseTypeDistribution::get_m(double rho, double prec)
{
	int mlo = -1;
	const double logprec = log(prec), pi = 3.141592653589793;
	double dmhi, dmlo;
	int mhi;

	dmhi = hifunc(rho, 0.0, logprec) - 1;
	dmlo = lofunc(rho, 2 * rho*h(dmhi / rho), logprec);
	if ((int)dmlo > mlo) {
		mlo = (int)dmlo;
	}

	if (log(boost::math::gamma_p((double)(mlo + 1), rho)) < logprec) {
		return mlo;
	}
	else {
		const double B = -0.5*log(4 * pi*rho*h(dmlo / rho));
		if (B > logprec) {
			dmhi = hifunc(rho, B, logprec);
		}
		mhi = (int)(dmhi + 1);

		while (mhi - mlo > 1) {
			int mmid = (mhi + mlo) / 2; // rounds down
			double dm = (double)mmid;
			double loginv;

			loginv = log(boost::math::gamma_p(dm + 1, rho));

			if (loginv < logprec) {
				mhi = mmid;
			}
			else {
				mlo = mmid;
			}
		}
	}
	return mhi;
}

//
// Sherlock, Chris. "Simple, Fast and Accurate Evaluation of the Action of the Exponential of a Rate Matrix on a Probability Vector."
// ArXiv : 1809.07110[Stat], September 19, 2018. http ://arxiv.org/abs/1809.07110.
//
RowVectorXd ShiftedPhaseTypeDistribution::vexpQ(RowVectorXd &v, SpMat &Q, double scaleQ)
{
	double prec = PREC;
	bool renorm = false;
	bool t2 = true;
	int *pm = NULL;
	const double rho = -scaleQ * (Q.diagonal().minCoeff()), BIG = 1e100;

	const int d = (int)v.cols(), nc = (int)v.rows(), mhi = get_m(rho, prec / (1.0 + (double)t2)), mlo = t2 ? get_mlo(mhi, rho) : 0;
	const double dnc = (double)nc;
	int j;

	// breaking up equations onto separate statements is faster with sparse matrices
	SpMat rMat(Q.rows(), Q.cols());
	rMat.setIdentity();
	rMat *= rho;

	SpMat P = Q;
	P *= scaleQ;
	P += rMat;

	RowVectorXd vsum(d), vpro = v;
	double fac = 1.0, final_mult, szvpro, rncumu = 0;

	szvpro = v.sum() / dnc;
	final_mult = szvpro;
	if (szvpro > BIG) {
		vpro /= szvpro;
		szvpro = 1; rncumu = log(szvpro);
	}
	if (mlo == 0) {
		vsum = vpro;
	}
	else {
		vsum = RowVectorXd::Zero(d);
	}

	j = 1;
	while (j <= mhi) { // do vsum <- v^T e^P
		vpro *= P;
		vpro /= fac;
		if (j >= mlo) {
			vsum += vpro;
		}
		szvpro *= rho / fac++;
		if (szvpro > BIG) {
			vpro /= szvpro;
			if (j >= mlo) {
				vsum /= szvpro;
			}
			if (!renorm) {
				rncumu += log(szvpro);
			}
			szvpro = 1;
		}
		j++;
	}

	if (pm != NULL) {
		*pm = mhi;
	}
	if (renorm) {
		final_mult /= vsum.sum() / dnc;
	}
	else {
		final_mult = exp(rncumu - rho);
	}

	return vsum * final_mult;
}

//
// Performs a row sum on a sparse matrix stored in the compressed column format
//
std::vector<double> ShiftedPhaseTypeDistribution::row_sum(SpMat* T)
{
	std::vector<double> result(T->cols());
	memset(&result[0], 0, sizeof(double) * T->cols());

	double* valuePtr = this->T->valuePtr();
	int* indexPtr = this->T->innerIndexPtr();

	for (register int i = (int)this->T->nonZeros(); i > 0; --i)
	{
		result[*indexPtr] += *valuePtr;

		++valuePtr;
		++indexPtr;
	}

	return result;
}
