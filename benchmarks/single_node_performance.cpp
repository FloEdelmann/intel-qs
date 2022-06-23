#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <vector>

// Include the IQS class.
#include "../include/qureg.hpp"

template <typename Type>
void benchmark(const std::string type_name, const int num_qubits, const int num_repetitions) {
  // Loop over the number of qubits and store the time elapsed in the computation.
  struct timeval time;
  double start, end;

  for (unsigned control_qubit = 0; control_qubit < num_qubits; control_qubit++) {
    for (unsigned target_qubit = 0; target_qubit < num_qubits; target_qubit++) {
      iqs::QubitRegister<Type> qubit_register(num_qubits, "++++");

      // MPI barrier and start the timer.
      iqs::mpi::StateBarrier();
      gettimeofday(&time, (struct timezone*)0);
      start = time.tv_sec + time.tv_usec * 1.0e-6;

      for (int iteration = 0; iteration < num_repetitions; iteration++) {
        if (control_qubit != target_qubit) {
          // qubit_register.ApplyCPauliX(control_qubit, target_qubit);
          qubit_register.ApplyCRotationY(control_qubit, target_qubit, 0.5);
        }
      }

      // MPI barrier and end the timer.
      iqs::mpi::StateBarrier();
      gettimeofday(&time, (struct timezone*)0);
      end = time.tv_sec + time.tv_usec * 1.0e-6;

      std::cout
        << type_name
        << ", " << control_qubit
        << ", " << target_qubit
        << ", " << ((end - start) / double(num_repetitions))
        << ", " << qubit_register.ComputeNorm()
        << "\n";
    }
  }
}

int main(int argc, char **argv) {
  iqs::mpi::Environment env(argc, argv);
  if (env.IsUsefulRank() == false) return 0;
  assert(env.GetNumStates() == 1);
  int my_rank = env.GetStateRank();
  int num_ranks = env.GetStateSize();

  int num_qubits = 10;
  int num_repetitions = 3;

  cout.precision(std::numeric_limits<float>::max_digits10);

  std::cout << "type, control_qubit, target_qubit, time_in_sec, l2_norm\n";

  benchmark<ComplexSP>(        "float32es5", num_qubits, num_repetitions);
  benchmark<ComplexPosit24es0>("posit24es0", num_qubits, num_repetitions);
  benchmark<ComplexPosit24es1>("posit24es1", num_qubits, num_repetitions);
  benchmark<ComplexPosit24es2>("posit24es2", num_qubits, num_repetitions);

  return 0;
}
