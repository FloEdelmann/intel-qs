#ifndef IQS_UTILS_HPP
#define IQS_UTILS_HPP

#include <complex>
#include <universal/number/posit/posit.hpp>

// Helpful defines, if not already provided.
#define DO_PRAGMA(x) _Pragma(#x)

#ifndef TODO
#define TODO(x) DO_PRAGMA(message("\033[30;43mTODO\033[0m - " #x))
#endif

#ifndef INFO
#define INFO(x) DO_PRAGMA(message("\033[30;46mINFO\033[0m - " #x))
#endif

#define UL(x) ((std::size_t)(x))
#define xstr(s) __str__(s)
#define __str__(s) #s

/////////////////////////////////////////////////////////////////////////////////////////

template<size_t es>
using IqsPosit24 = sw::universal::posit<24, es>;

using IqsPosit24es0 = IqsPosit24<0>;
using IqsPosit24es1 = IqsPosit24<1>;
using IqsPosit24es2 = IqsPosit24<2>;

using ComplexSP = std::complex<float>;
using ComplexDP = std::complex<double>;

template<size_t es>
using ComplexPosit24 = std::complex<IqsPosit24<es>>;

using ComplexPosit24es0 = ComplexPosit24<0>;
using ComplexPosit24es1 = ComplexPosit24<1>;
using ComplexPosit24es2 = ComplexPosit24<2>;

namespace iqs {

/////////////////////////////////////////////////////////////////////////////////////////

// Structure to extract the value type of a template.
template<typename T>
struct extract_value_type
{
    typedef T value_type;
};

// Structure to extract the value type of a template of template.
template<template<typename> class X, typename T>
struct extract_value_type<X<T>>   //specialization
{
    typedef T value_type;
};

/////////////////////////////////////////////////////////////////////////////////////////

double time_in_seconds(void);

/////////////////////////////////////////////////////////////////////////////////////////

/// Utility method to inform on the currently set compiler flags.
void WhatCompileDefinitions();

/////////////////////////////////////////////////////////////////////////////////////////

} // end namespace iqs

#endif	// header guard IQS_UTILS_HPP
