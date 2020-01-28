//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information. 
//
// Written by Luke Marshall and Timur Tankayev, 2019
//

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#include <windows.h>

#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <numpy/ndarrayobject.h>

#include "..\ShiftedPhaseType\ShiftedPhaseType.h"
#include "..\ShiftedPhaseType\DistanceLookup.h"

using namespace ShiftedPhaseType;

// Adapter a member function that returns a unique_ptr to a python function object that 
// returns a raw pointer but explicitly passes ownership to Python.
template <typename T, typename C, typename ...Args>
boost::python::object adapt_unique(std::unique_ptr<T>(C::* fn)(Args...))
{
	return boost::python::make_function(
		[fn](C& self, Args... args) { return (self.*fn)(args...).release(); },
		boost::python::return_value_policy<boost::python::manage_new_object>(),
		boost::mpl::vector<T*, C&, Args...>()
	);
}

// Convert std::tuple<T1,T2> to python tuple
template<typename T1, typename T2>
struct std_tuple_to_python_tuple {
	static PyObject* convert(const std::tuple<T1, T2>& t) {
		return boost::python::incref(
			boost::python::object(
				boost::python::make_tuple(std::get<0>(t), std::get<1>(t))).ptr());
	}
};

// Convert std::vector<T> to a python list
template<typename T>
struct std_vector_to_python_list {
	static PyObject* convert(const std::vector<T>& coll) {
		boost::python::list list;

		for (const auto& t : coll) {
			list.append(t);
		}

		return boost::python::incref(boost::python::object(list).ptr());
	}
};

// Specialized wrapper functions for ShiftedPhaseTypeDistribution class, in particular it converts RowVectorXd and SpMat to/from python
class WrapperFuncs
{
public:
	// Constructor wrapper
	static boost::shared_ptr<ShiftedPhaseTypeDistribution> initWrapper(double shift, boost::python::object const& alpha_py, boost::python::object const& t_py)
	{
		if (!PyArray_CheckExact(alpha_py.ptr())) {
			throw std::runtime_error("The given alpha is not a valid array");
		}

		if (!PyArray_CheckExact(t_py.ptr())) {
			throw std::runtime_error("The given T is not a valid array");
		}

		if (PyArray_TYPE(alpha_py.ptr()) != NPY_DOUBLE) {
			throw std::runtime_error("The alpha array should have double precision");
		}

		if (PyArray_TYPE(t_py.ptr()) != NPY_DOUBLE) {
			throw std::runtime_error("The T matrix should have double precision");
		}


		// We assume that there is only one dimension and assume double precision
		int size = (int)PyArray_SHAPE((PyArrayObject*)alpha_py.ptr())[0];

		RowVectorXd alpha(size);
		memcpy(alpha.data(), (double*)PyArray_DATA(alpha_py.ptr()), size * sizeof(double));

		// Assume dense matrix, only two dimensions, double precision
		int rows = (int)PyArray_SHAPE((PyArrayObject*)t_py.ptr())[0];
		int cols = (int)PyArray_SHAPE((PyArrayObject*)t_py.ptr())[1];

		SpMat t(rows, cols);

		// Count non-zero entries
		double* p = (double*)PyArray_DATA(t_py.ptr());

		int nnz = 0;

		for (int j = 0; j < cols; ++j) {
			for (int i = 0; i < rows; ++i) {
				if (p[i * rows + j] != 0.0) {
					++nnz;
				}
			}
		}

		// Load sparse matrix from dense numpy array
		SparseLoad loader(&t, nnz);

		for (int j = 0; j < cols; ++j) {
			for (int i = 0; i < rows; ++i) {
				if (p[i * rows + j] != 0.0) {
					loader.add(p[i * rows + j], i);
				}
			}
			loader.next_col();
		}

		return boost::shared_ptr<ShiftedPhaseTypeDistribution>(new ShiftedPhaseTypeDistribution(shift, alpha, t));
	}

	// Returns copy of Alpha vector
	static boost::python::object ShiftedPhaseType_Alpha(ShiftedPhaseTypeDistribution* d)
	{
		npy_intp shape[1] = { d->Alpha->size() };
		return boost::python::object(boost::python::handle<>(PyArray_SimpleNewFromData(1, shape, NPY_DOUBLE, (double*)d->Alpha->data())));
	}

	// Returns copy of T matrix (converted to dense format)
	static boost::python::object ShiftedPhaseType_T(ShiftedPhaseTypeDistribution* d) {
		npy_intp shape[2] = { d->T->rows(), d->T->cols() };
		PyObject* matrix = PyArray_SimpleNew(2, shape, NPY_DOUBLE);

		double* p = (double*)PyArray_DATA(matrix);

		for (int i = 0; i < d->T->rows(); ++i) {
			for (int j = 0; j < d->T->cols(); ++j) {
				p[i * d->T->rows() + j] = d->T->coeff(i, j);
			}
		}

		return boost::python::object(boost::python::handle<>(matrix));
	}
};

BOOST_PYTHON_MODULE(ShiftedPhaseTypePy)
{
	_import_array();	// numpy requires this

	using namespace boost::python;

	boost::python::to_python_converter<std::tuple<std::vector<double>, std::vector<double>>, std_tuple_to_python_tuple<std::vector<double>, std::vector<double>>>();
	boost::python::to_python_converter<std::tuple<double, double>, std_tuple_to_python_tuple<double, double>>();
	boost::python::to_python_converter<std::vector<double>, std_vector_to_python_list<double>>();

	class_<DistanceLookup>("DistanceLookup", init<std::string>())
		.def("GetDistance", &DistanceLookup::GetDistance)
		.def("FitDistance", &DistanceLookup::FitDistance)
		.def("GetProfile", &DistanceLookup::GetProfile);

	std::unique_ptr<ShiftedPhaseTypeDistribution>(ShiftedPhaseTypeDistribution:: * add1)(double) = &ShiftedPhaseTypeDistribution::Add;
	std::unique_ptr<ShiftedPhaseTypeDistribution>(ShiftedPhaseTypeDistribution:: * add2)(ShiftedPhaseTypeDistribution&, double) = &ShiftedPhaseTypeDistribution::Add;

	class_<ShiftedPhaseTypeDistribution>("ShiftedPhaseTypeDistribution", no_init)
		// Constructors
		.def("__init__", boost::python::make_constructor(&WrapperFuncs::initWrapper))
		.def(init<optional<double>>())
		.def(init<double, double, double>())

		// Member variables
		.def_readonly("Shift", &ShiftedPhaseTypeDistribution::Shift)
		.def_readonly("GreaterThanConditionalProbability", &ShiftedPhaseTypeDistribution::GreaterThanConditionalProbability)
		.def_readonly("GreaterThanConditionalScale", &ShiftedPhaseTypeDistribution::GreaterThanConditionalScale)
		.add_property("Alpha", &WrapperFuncs::ShiftedPhaseType_Alpha)
		.add_property("T", &WrapperFuncs::ShiftedPhaseType_T)

		// Add/Max operators
		.def("Add", adapt_unique(add1))
		.def("Add", adapt_unique(add2))
		.def("ShiftedAndScaledPH", adapt_unique(&ShiftedPhaseTypeDistribution::ShiftedAndScaledPH))
		.def("MaxWith", adapt_unique(&ShiftedPhaseTypeDistribution::MaxWith))
		.def("MaxWithAdd", adapt_unique(&ShiftedPhaseTypeDistribution::MaxWithAdd))
		.def("LinearMapWithMax", adapt_unique(&ShiftedPhaseTypeDistribution::LinearMapWithMax))

		// Member properties
		.def("IsPointMass", &ShiftedPhaseTypeDistribution::IsPointMass)
		.def("Density", &ShiftedPhaseTypeDistribution::Density)
		.def("CumulativeDistribution", &ShiftedPhaseTypeDistribution::CumulativeDistribution)
		.def("Unshifted1Moment", &ShiftedPhaseTypeDistribution::Unshifted1Moment)
		.def("Unshifted2Moments", &ShiftedPhaseTypeDistribution::Unshifted2Moments)
		.def("Sample", &ShiftedPhaseTypeDistribution::Sample)

		// Truncation operators
		.def("LeftTruncMoments", &ShiftedPhaseTypeDistribution::LeftTruncMoments)
		.def("RightTruncMoments", &ShiftedPhaseTypeDistribution::RightTruncMoments)
		.def("RightExpProb", &ShiftedPhaseTypeDistribution::RightExpProb)
		.def("ShiftedAndScaledPH", adapt_unique(&ShiftedPhaseTypeDistribution::ShiftedAndScaledPH))
		.def("ConditionGreaterThan", adapt_unique(&ShiftedPhaseTypeDistribution::ConditionGreaterThan));
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}
