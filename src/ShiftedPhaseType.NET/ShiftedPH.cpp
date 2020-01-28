//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information. 
//
// Written by Luke Marshall and Timur Tankayev, 2019
//
#include "ShiftedPH.h"
#include "..\ShiftedPhaseType\SparseLoad.h"

using namespace ShiftedPhaseType::NET;

ShiftedPH::ShiftedPH(array<double>^ erlangs)
{
	int N = erlangs->Length;

	int nPhases = 0;
	for (int k = 0; k < N; k += 3)
	{
		nPhases += (int)erlangs[k];
	}

	if (nPhases == 0)
		shiftedPhaseType = new ShiftedPhaseTypeDistribution();
	else
	{
		RowVectorXd alpha(nPhases);
		alpha.fill(0.0);

		// fast load sparse matrix in compressed column format
		SpMat T(nPhases, nPhases);
		SparseLoad loader(&T, 2 * nPhases - N / 3);

		int i = 0;
		for (int k = 0; k < N; k += 3)
		{
			int n = (int)erlangs[k];
			double lambda = erlangs[k + 1];

			alpha(i) = erlangs[k + 2];

			for (int j = 0; j < n - 1; ++j)
			{
				if (j > 0)
				{
					loader.add(lambda, i - 1);
				}

				loader.add(-lambda, i);
				loader.next_col();

				i++;
			}

			if (i > 0)
			{
				loader.add(lambda, i - 1);
			}

			loader.add(-lambda, i);
			loader.next_col();

			++i;
		}

		shiftedPhaseType = new ShiftedPhaseTypeDistribution(0, alpha, T);
	}
}
