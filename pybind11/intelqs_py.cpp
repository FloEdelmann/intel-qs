#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <vector>

#include "../include/qureg.hpp"
#include "../include/mpi_env.hpp"
#include "../include/rng_utils.hpp"

// Extra feature. It can be included optionally.
#if 1
#include "../include/qaoa_features.hpp"
#endif

#ifdef INTELQS_HAS_MPI
#include <mpi.h>
#endif

//////////////////////////////////////////////////////////////////////////////

namespace py = pybind11;
using Environment = iqs::mpi::Environment;

namespace iqs {

//////////////////////////////////////////////////////////////////////////////

void EnvInit()
{ Environment::Init(); }

void EnvFinalize()
{ Environment::Finalize(); }

void EnvFinalizeDummyRanks()
{
  if (Environment::GetSharedInstance()->IsUsefulRank()==false)
  {
      Environment::Finalize();
      return;
  }
}

//////////////////////////////////////////////////////////////////////////////
// PYBIND CODE for the Intel Quantum Simulator library
//////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(intelqs_py, m)
{
    m.doc() = "pybind11 wrap for the Intel Quantum Simulator";


//////////////////////////////////////////////////////////////////////////////
// Init & Finalize for HPC
//////////////////////////////////////////////////////////////////////////////
    m.def("EnvInit", &EnvInit, "Initialize the MPI environment of Intel-QS for HPC resource allocation");
    m.def("EnvFinalize", &EnvFinalize, "Finalize the MPI environemtn fo Intel-QS");
    m.def("EnvFinalizeDummyRanks", &EnvFinalizeDummyRanks, "Finalize the dummy ranks of the MPI environment");

//////////////////////////////////////////////////////////////////////////////
// Utilities
//////////////////////////////////////////////////////////////////////////////

    // Random Number Generator
    py::class_<iqs::RandomNumberGenerator<float>>(m, "RandomNumberGenerator")
        .def(py::init<>())
        .def("GetSeed", &iqs::RandomNumberGenerator<float>::GetSeed)
        .def("SetSeedStreamPtrs", &iqs::RandomNumberGenerator<float>::SetSeedStreamPtrs)
        .def("SkipeAhead", &iqs::RandomNumberGenerator<float>::SkipAhead)
        .def("UniformRandomNumbers", &iqs::RandomNumberGenerator<float>::UniformRandomNumbers)
        .def("GaussianRandomNumbers", &iqs::RandomNumberGenerator<float>::GaussianRandomNumbers)
        .def("RandomIntegersInRange", &iqs::RandomNumberGenerator<float>::RandomIntegersInRange)
        .def("GetUniformRandomNumbers",
             [](iqs::RandomNumberGenerator<float> &rng, std::size_t size,
                float a, float b, std::string shared) {
                std::vector<float> random_values(size);
                rng.UniformRandomNumbers(random_values.data(), size, a, b, shared);
                return random_values;
             }, "Return an array of 'size' random number from the uniform distribution [a,b[.")
#ifdef WITH_MPI_AND_MKL
        .def("SetRndStreamPtrs", &iqs::RandomNumberGenerator<float>::SetRndStreamPtrs)
#endif
        .def("__repr__", []() { return "<RandomNumberGenerator specialized for MKL.>"; } );


    // Chi Matrix 4x4
    py::class_<iqs::ChiMatrix<ComplexSP, 4, 32>>(m, "CM4x4", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<>())
        // Access element:
        .def("__getitem__", [](const iqs::ChiMatrix<ComplexSP,4,32> &a, std::pair<py::ssize_t, py::ssize_t> i, int column) {
             if (i.first > 4) throw py::index_error();
             if (i.second > 4) throw py::index_error();
std::cout << "ciao\n";
             return a(i.first, i.second);
             }, py::is_operator())
        // Set element:
        .def("__setitem__", [](iqs::ChiMatrix<ComplexSP,4,32> &a, std::pair<py::ssize_t, py::ssize_t> i, ComplexSP value) {
             if (i.first > 4) throw py::index_error();
             if (i.second > 4) throw py::index_error();
             a(i.first, i.second) = value;
             }, py::is_operator())
#if 0
        .def_buffer([](iqs::ChiMatrix<ComplexSP,4,32> &m) -> py::buffer_info {
            return py::buffer_info(
                m.GetPtrToData(),                      /* Pointer to buffer */
                sizeof(ComplexSP),                     /* Size of one scalar */
                py::format_descriptor<ComplexSP>::format(), /* Python struct-style format descriptor */
                std::size_t(2),                                      /* Number of dimensions */
                { 4, 4 },                 /* Buffer dimensions */
                { 4*sizeof(ComplexSP), 4 });             /* Strides (in bytes) for each index */
        })
#endif
#if 0
        .def("ApplyChannel",
             [](QubitRegister<ComplexSP> &a, unsigned qubit,
                py::array_t<ComplexSP, py::array::c_style | py::array::forcecast> matrix ) {
               py::buffer_info buf = matrix.request();
               if (buf.ndim != 2)
                   throw std::runtime_error("Number of dimensions must be two.");
               if (buf.shape[0] != 4 || buf.shape[1] != 4)
                   throw std::runtime_error("The shape of the chi-matrix is not 4x4.");
               // Create and initialize the custom chi-matrix used by Intel QS.
               ComplexSP *ptr = (ComplexSP *) buf.ptr;
               CM4x4<ComplexSP> m;
               m(0,0)=ptr[0];  m(0,1)=ptr[1];  m(0,2)=ptr[2];  m(0,3)=ptr[3];
               m(1,0)=ptr[4];  m(1,1)=ptr[5];  m(1,2)=ptr[6];  m(1,3)=ptr[7];
               m(2,0)=ptr[8];  m(2,1)=ptr[9];  m(2,2)=ptr[10]; m(2,3)=ptr[11];
               m(3,0)=ptr[12]; m(3,1)=ptr[13]; m(3,2)=ptr[14]; m(3,3)=ptr[15];
               a.ApplyChannel(qubit, m);
             }, "Apply 1-qubit channel provided via its chi-matrix.")
#endif
        .def("SolveEigenSystem", &iqs::ChiMatrix<ComplexSP,4,32>::SolveEigenSystem)
        .def("Print", &iqs::ChiMatrix<ComplexSP,4,32>::Print)
        .def("__repr__", []() { return "<ChiMatrix for 1-qubit channel>"; } );

    // Chi Matrix 16x16
    py::class_<iqs::ChiMatrix<ComplexSP,16,32>>(m, "CM16x16")
        .def(py::init<>())
        .def(py::init<>())
        // Access element:
        .def("__getitem__", [](const iqs::ChiMatrix<ComplexSP,16,32> &a, std::pair<py::ssize_t, py::ssize_t> i, int column) {
             if (i.first > 16) throw py::index_error();
             if (i.second > 16) throw py::index_error();
             return a(i.first, i.second);
             }, py::is_operator())
        // Set element:
        .def("__setitem__", [](iqs::ChiMatrix<ComplexSP,16,32> &a, std::pair<py::ssize_t, py::ssize_t> i, ComplexSP value) {
             if (i.first > 16) throw py::index_error();
             if (i.second > 16) throw py::index_error();
             a(i.first, i.second) = value;
             }, py::is_operator())
        .def("SolveEigenSystem", &iqs::ChiMatrix<ComplexSP,16,32>::SolveEigenSystem)
        .def("Print", &iqs::ChiMatrix<ComplexSP,16,32>::Print)
        .def("__repr__", []() { return "<ChiMatrix for 2-qubit channel>"; } );

//////////////////////////////////////////////////////////////////////////////
// Intel-QS
//////////////////////////////////////////////////////////////////////////////

    // Intel Quantum Simulator
    // Notice that to use std::cout in the C++ code, one needs to redirect the output streams.
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html
//    py::class_<QubitRegister<ComplexSP>, shared_ptr< QubitRegister<ComplexSP> >>(m, "QubitRegister")
    py::class_< QubitRegister<ComplexSP> >(m, "QubitRegister", py::buffer_protocol(), py::dynamic_attr())
        .def(py::init<> ())
        .def(py::init<const QubitRegister<ComplexSP> &>())	// copy constructor
        .def(py::init<std::size_t , std::string , std::size_t, std::size_t> ())
        // Information on the internal representation:
        .def("NumQubits", &QubitRegister<ComplexSP>::NumQubits)
        .def("GlobalSize", &QubitRegister<ComplexSP>::GlobalSize)
        .def("LocalSize" , &QubitRegister<ComplexSP>::LocalSize )
        // Access element:
        .def("__getitem__", [](const QubitRegister<ComplexSP> &a, std::size_t index) {
             if (index >= a.LocalSize()) throw py::index_error();
             return a[index];
             }, py::is_operator())
        // Set element:
        .def("__setitem__", [](QubitRegister<ComplexSP> &a, std::size_t index, ComplexSP value) {
             if (index >= a.LocalSize()) throw py::index_error();
             a[index] = value;
             }, py::is_operator())
        // Numpy buffer protocol
        // See https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
        .def_buffer([](QubitRegister<ComplexSP> &reg) -> py::buffer_info {
            return py::buffer_info(
                reg.RawState(),                            /* Pointer to buffer */
                sizeof(ComplexSP),                          /* Size of one scalar */
                py::format_descriptor<ComplexSP>::format(), /* Python struct-style format descriptor */
                1,                                      /* Number of dimensions */
                { reg.LocalSize() },                 /* Buffer dimensions */
                { sizeof(ComplexSP) });             /* Strides (in bytes) for each index */
        })
        // One-qubit gates:
        .def("ApplyRotationX", &QubitRegister<ComplexSP>::ApplyRotationX)
        .def("ApplyRotationY", &QubitRegister<ComplexSP>::ApplyRotationY)
        .def("ApplyRotationZ", &QubitRegister<ComplexSP>::ApplyRotationZ)
        .def("ApplyPauliX", &QubitRegister<ComplexSP>::ApplyPauliX)
        .def("ApplyPauliY", &QubitRegister<ComplexSP>::ApplyPauliY)
        .def("ApplyPauliZ", &QubitRegister<ComplexSP>::ApplyPauliZ)
        .def("ApplyPauliSqrtX", &QubitRegister<ComplexSP>::ApplyPauliSqrtX)
        .def("ApplyPauliSqrtY", &QubitRegister<ComplexSP>::ApplyPauliSqrtY)
        .def("ApplyPauliSqrtZ", &QubitRegister<ComplexSP>::ApplyPauliSqrtZ)
        .def("ApplyT", &QubitRegister<ComplexSP>::ApplyT)
        .def("ApplyHadamard", &QubitRegister<ComplexSP>::ApplyHadamard)
        // Two-qubit gates:
        .def("ApplySwap", &QubitRegister<ComplexSP>::ApplySwap)
        .def("ApplyCRotationX", &QubitRegister<ComplexSP>::ApplyCRotationX)
        .def("ApplyCRotationY", &QubitRegister<ComplexSP>::ApplyCRotationY)
        .def("ApplyCRotationZ", &QubitRegister<ComplexSP>::ApplyCRotationZ)
        .def("ApplyCPauliX", &QubitRegister<ComplexSP>::ApplyCPauliX)
        .def("ApplyCPauliY", &QubitRegister<ComplexSP>::ApplyCPauliY)
        .def("ApplyCPauliZ", &QubitRegister<ComplexSP>::ApplyCPauliZ)
        .def("ApplyCPauliSqrtZ", &QubitRegister<ComplexSP>::ApplyCPauliSqrtZ)
        .def("ApplyCHadamard", &QubitRegister<ComplexSP>::ApplyCHadamard)
        // Custom 1-qubit gate and controlled 2-qubit gates:
        .def("Apply1QubitGate",
             [](QubitRegister<ComplexSP> &a, unsigned qubit,
                py::array_t<ComplexSP, py::array::c_style | py::array::forcecast> matrix ) {
               py::buffer_info buf = matrix.request();
               if (buf.ndim != 2)
                   throw std::runtime_error("Number of dimensions must be two.");
               if (buf.shape[0] != 2 || buf.shape[1] != 2)
                   throw std::runtime_error("Input shape is not 2x2.");
               // Create and initialize the custom tiny-matrix used by Intel QS.
               ComplexSP *ptr = (ComplexSP *) buf.ptr;
               TM2x2<ComplexSP> m;
               m(0,0)=ptr[0];
               m(0,1)=ptr[1];
               m(1,0)=ptr[2];
               m(1,1)=ptr[3];
               a.Apply1QubitGate(qubit, m);
             }, "Apply custom 1-qubit gate.")
        .def("ApplyControlled1QubitGate",
             [](QubitRegister<ComplexSP> &a, unsigned control, unsigned qubit,
                py::array_t<ComplexSP, py::array::c_style | py::array::forcecast> matrix ) {
               py::buffer_info buf = matrix.request();
               if (buf.ndim != 2)
                   throw std::runtime_error("Number of dimensions must be two.");
               if (buf.shape[0] != 2 || buf.shape[1] != 2)
                   throw std::runtime_error("The shape of the unitary-matrix is not 2x2.");
               // Create and initialize the custom tiny-matrix used by Intel QS.
               ComplexSP *ptr = (ComplexSP *) buf.ptr;
               TM2x2<ComplexSP> m;
               m(0,0)=ptr[0];
               m(0,1)=ptr[1];
               m(1,0)=ptr[2];
               m(1,1)=ptr[3];
               a.ApplyControlled1QubitGate(control, qubit, m);
             }, "Apply custom controlled-1-qubit gate.")
        // Apply 1-qubit and 2-qubit channel.
        .def("GetOverallSignOfChannels", &QubitRegister<ComplexSP>::GetOverallSignOfChannels)
#if 1
        .def("ApplyChannel",
             [](QubitRegister<ComplexSP> &a, unsigned qubit, iqs::ChiMatrix<ComplexSP,4,32> chi) {
               a.ApplyChannel(qubit, chi);
             }, "Apply 1-qubit channel provided via its chi-matrix.")
        .def("ApplyChannel",
             [](QubitRegister<ComplexSP> &a, unsigned qubit1, unsigned qubit2,
                iqs::ChiMatrix<ComplexSP,16,32> chi) {
               a.ApplyChannel(qubit1, qubit2, chi);
             }, "Apply 2-qubit channel provided via its chi-matrix.")
#else
        .def("ApplyChannel",
             [](QubitRegister<ComplexSP> &a, unsigned qubit,
                py::array_t<ComplexSP, py::array::c_style | py::array::forcecast> matrix ) {
               py::buffer_info buf = matrix.request();
               if (buf.ndim != 2)
                   throw std::runtime_error("Number of dimensions must be two.");
               if (buf.shape[0] != 4 || buf.shape[1] != 4)
                   throw std::runtime_error("The shape of the chi-matrix is not 4x4.");
               // Create and initialize the custom chi-matrix used by Intel QS.
               ComplexSP *ptr = (ComplexSP *) buf.ptr;
               CM4x4<ComplexSP> m;
               m(0,0)=ptr[0];  m(0,1)=ptr[1];  m(0,2)=ptr[2];  m(0,3)=ptr[3];
               m(1,0)=ptr[4];  m(1,1)=ptr[5];  m(1,2)=ptr[6];  m(1,3)=ptr[7];
               m(2,0)=ptr[8];  m(2,1)=ptr[9];  m(2,2)=ptr[10]; m(2,3)=ptr[11];
               m(3,0)=ptr[12]; m(3,1)=ptr[13]; m(3,2)=ptr[14]; m(3,3)=ptr[15];
               a.ApplyChannel(qubit, m);
             }, "Apply 1-qubit channel provided via its chi-matrix.")
        .def("ApplyChannel",
             [](QubitRegister<ComplexSP> &a, unsigned qubit1, unsigned qubit2,
                py::array_t<ComplexSP, py::array::c_style | py::array::forcecast> matrix ) {
               py::buffer_info buf = matrix.request();
               if (buf.ndim != 2)
                   throw std::runtime_error("Number of dimensions must be two.");
               if (buf.shape[0] != 16 || buf.shape[1] != 16)
                   throw std::runtime_error("The shape of the chi-matrix is not 16x16.");
               // Create and initialize the custom chi-matrix used by Intel QS.
               ComplexSP *ptr = (ComplexSP *) buf.ptr;
               CM16x16<ComplexSP> m;
               int index = 0;
               for (int i=0; i<16; ++i)
               for (int j=0; j<16; ++j)
               {
                   m(i,j)=ptr[index];
                   index += 1;
               }
               a.ApplyChannel(qubit1, qubit2, m);
             }, "Apply 1-qubit channel provided via its chi-matrix.")
#endif
        // Three-qubit gates:
        .def("ApplyToffoli", &QubitRegister<ComplexSP>::ApplyToffoli)
        // State initialization:
        .def("Initialize",
               (void (QubitRegister<ComplexSP>::*)(std::string, std::size_t ))
                 &QubitRegister<ComplexSP>::Initialize)
        //Enable Specialization
        .def("TurnOnSpecialize", &QubitRegister<ComplexSP>::TurnOnSpecialize)
        .def("TurnOffSpecialize", &QubitRegister<ComplexSP>::TurnOffSpecialize)
        .def("TurnOnSpecializeV2", &QubitRegister<ComplexSP>::TurnOnSpecializeV2)
        .def("TurnOffSpecializeV2", &QubitRegister<ComplexSP>::TurnOffSpecializeV2)
        // Associate the random number generator and set its seed.
        .def("ResetRngPtr", &QubitRegister<ComplexSP>::ResetRngPtr)
        .def("SetRngPtr", &QubitRegister<ComplexSP>::SetRngPtr)
        .def("SetSeedRngPtr", &QubitRegister<ComplexSP>::SetSeedRngPtr)
        // State measurement and collapse:
        .def("GetProbability", &QubitRegister<ComplexSP>::GetProbability)
        .def("CollapseQubit", &QubitRegister<ComplexSP>::CollapseQubit)
          // Recall that the collapse selects: 'false'=|0> , 'true'=|1>
        .def("Normalize", &QubitRegister<ComplexSP>::Normalize)
        .def("ExpectationValue", &QubitRegister<ComplexSP>::ExpectationValue)
        // Other quantum operations:
        .def("ComputeNorm", &QubitRegister<ComplexSP>::ComputeNorm)
        .def("ComputeOverlap", &QubitRegister<ComplexSP>::ComputeOverlap)
        // Noisy simulation
        .def("GetT1", &QubitRegister<ComplexSP>::GetT1)
        .def("GetT2", &QubitRegister<ComplexSP>::GetT2)
        .def("GetTphi", &QubitRegister<ComplexSP>::GetTphi)
        .def("SetNoiseTimescales", &QubitRegister<ComplexSP>::SetNoiseTimescales)
        .def("ApplyNoiseGate", &QubitRegister<ComplexSP>::ApplyNoiseGate)
        // Utility functions:
        .def("Print",
             [](QubitRegister<ComplexSP> &a, std::string description) {
               py::scoped_ostream_redirect stream(
               std::cout,                               // std::ostream&
               py::module::import("sys").attr("stdout") // Python output
               );
               std::vector<size_t> qubits = {};
               std::cout << "<<the output has been redirected to the terminal>>\n";
               a.Print(description, qubits);
             }, "Print the quantum state with an initial description.");


//////////////////////////////////////////////////////////////////////////////
// Extra features: QAOA circuits
//////////////////////////////////////////////////////////////////////////////

#ifdef QAOA_EXTRA_FEATURES_HPP
    m.def("InitializeVectorAsMaxCutCostFunction",
          &qaoa::InitializeVectorAsMaxCutCostFunction<ComplexSP>,
          "Use IQS vector to store a large real vector and not as a quantum state.");

    m.def("InitializeVectorAsWeightedMaxCutCostFunction",
          &qaoa::InitializeVectorAsWeightedMaxCutCostFunction<ComplexSP>,
          "Use IQS vector to store a large real vector and not as a quantum state.");

    m.def("ImplementQaoaLayerBasedOnCostFunction",
          &qaoa::ImplementQaoaLayerBasedOnCostFunction<ComplexSP>,
          "Implement exp(-i gamma C)|psi>.");

    m.def("GetExpectationValueFromCostFunction",
          &qaoa::GetExpectationValueFromCostFunction<ComplexSP>,
          "Get expectation value from the cost function.");

    m.def("GetExpectationValueSquaredFromCostFunction",
          &qaoa::GetExpectationValueSquaredFromCostFunction<ComplexSP>,
          "Get expectation value squared from the cost function.");

    m.def("GetHistogramFromCostFunction",
          &qaoa::GetHistogramFromCostFunction<ComplexSP>,
          "Get histogram instead of just the expectation value.");
        
    m.def("GetHistogramFromCostFunctionWithWeightsRounded",
          &qaoa::GetHistogramFromCostFunctionWithWeightsRounded<ComplexSP>,
          "Get histogram instead of just the expectation value for a weighted graph, with all cut values rounded down.");
    
    m.def("GetHistogramFromCostFunctionWithWeightsBinned",
          &qaoa::GetHistogramFromCostFunctionWithWeightsBinned<ComplexSP>,
          "Get histogram instead of just the expectation value for a weighted graph, with specified bin width.");
#endif


//////////////////////////////////////////////////////////////////////////////
// MPI Features
//////////////////////////////////////////////////////////////////////////////
    py::class_<Environment>(m, "MPIEnvironment")
        .def(py::init<>())
        .def_static("GetRank", &Environment::GetRank)
        .def_static("IsUsefulRank", &Environment::IsUsefulRank)
        .def_static("GetSizeWorldComm",
             []() {
               int world_size = 1;
#ifdef INTELQS_HAS_MPI
               MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif
               return world_size;
             }, "Number of processes when the MPI environment was first created.")
        .def_static("GetPoolRank", &Environment::GetPoolRank)
        .def_static("GetStateRank", &Environment::GetStateRank)
        .def_static("GetPoolSize", &Environment::GetPoolSize)
        .def_static("GetStateSize", &Environment::GetStateSize)

        .def_static("GetNumRanksPerNode", &Environment::GetNumRanksPerNode)
        .def_static("GetNumNodes", &Environment::GetNumNodes)
        .def_static("GetStateId", &Environment::GetStateId)
        .def_static("GetNumStates", &Environment::GetNumStates)

        .def_static("Barrier", &iqs::mpi::Barrier)
        .def_static("PoolBarrier", &iqs::mpi::PoolBarrier)
        .def_static("StateBarrier", &iqs::mpi::StateBarrier)

        .def_static("IncoherentSumOverAllStatesOfPool", &Environment::IncoherentSumOverAllStatesOfPool<float>)
        .def_static("UpdateStateComm", &Environment::UpdateStateComm);

}

//////////////////////////////////////////////////////////////////////////////

} // end namespace iqs

