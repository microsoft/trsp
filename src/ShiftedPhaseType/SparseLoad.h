//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information. 
//
// Written by Luke Marshall and Timur Tankayev, 2019
//
#pragma once

#pragma warning(push, 0)        
#include <Eigen/Sparse>
#pragma warning(pop)

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> tripl;

//
// Directly constructs an Eigen sparse matrix in compressed column format.
// Assumes that data is loaded in order (by column then row)
//
class SparseLoad
{
	double *valuePtr;
	int *innerPtr;
	int *outerPtr;

	int nnzCount;
	int num_cols;

public:
	SparseLoad(SpMat *m, size_t nnz)
	{
		m->resizeNonZeros(nnz);

		valuePtr = m->valuePtr();
		innerPtr = m->innerIndexPtr();
		outerPtr = m->outerIndexPtr();
		num_cols = (int) m->cols() + 1;
		nnzCount = 0;

		next_col(); // initialize first entry to zero
	}

	inline void add(double v, size_t index)
	{
		*valuePtr = v;
		*innerPtr = (int)index;

		++valuePtr;
		++innerPtr;
		++nnzCount;
	}

	inline void next_col()
	{
		assert(--num_cols >= 0);
		*outerPtr = nnzCount;
		++outerPtr;
	}

	void initial_block(SpMat *T)
	{
		// to add support for additional blocks, need to:
		//  * track #rows in last block (or specify index)
		//	* add index to T->innerIndexPtr
		//	* add this->count to T->outerIndexPtr

		Eigen::Index nnz = T->nonZeros();
		memcpy(valuePtr, T->valuePtr(), nnz * sizeof(double));
		memcpy(innerPtr, T->innerIndexPtr(), nnz * sizeof(int));

		memcpy(--outerPtr, T->outerIndexPtr(), T->cols() * sizeof(int));

		valuePtr += nnz;
		innerPtr += nnz;
		nnzCount += (int)nnz;

		outerPtr += T->cols();
		num_cols -= (int) T->cols() - 1;
		assert(num_cols >= 0);
	}
};
