
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <vector>

// Include the IQS class.
#include "../include/qureg.hpp"

int main(int argc, char **argv)
{
  iqs::mpi::Environment env(argc, argv);
  if (env.IsUsefulRank() == false) return 0;
  assert(env.GetNumStates() == 1);
  int my_rank = env.GetStateRank();
  int num_ranks = env.GetStateSize();

  int num_qubits = 29;

  int num_repetitions = 3;

/////////////////////////////////////////////////////////////////////////////////////////

  // Loop over the number of qubits and store the time elapsed in the computation.
  struct timeval time;
  double start, end;

  std::cout << "type, control_qubit, target_qubit, time_in_sec\n";

  for (unsigned control_qubit = 0; control_qubit < num_qubits; control_qubit++) {
    for (unsigned target_qubit = 0; target_qubit < num_qubits; target_qubit++) {
      iqs::QubitRegister<ComplexDP> qubit_register(num_qubits, "base", 0);

      // MPI barrier and start the timer.
      iqs::mpi::StateBarrier();
      gettimeofday(&time, (struct timezone*)0);
      start = time.tv_sec + time.tv_usec * 1.0e-6;

      for (int iteration = 0; iteration < num_repetitions; iteration++) {
        if (control_qubit != target_qubit) {
          qubit_register.ApplyCPauliX(control_qubit, target_qubit);
        }
      }

      // MPI barrier and end the timer.
      iqs::mpi::StateBarrier();
      gettimeofday(&time, (struct timezone*)0);
      end = time.tv_sec + time.tv_usec * 1.0e-6;

      std::cout << "ComplexDP, " << control_qubit << ", " << target_qubit << ", " << ((end - start) / double(num_repetitions)) << "\n";
    }
  }

/////////////////////////////////////////////////////////////////////////////////////////

  for (unsigned control_qubit = 0; control_qubit < num_qubits; control_qubit++) {
    for (unsigned target_qubit = 0; target_qubit < num_qubits; target_qubit++) {
      iqs::QubitRegister<ComplexPosit> qubit_register(num_qubits, "base", 0);

      // MPI barrier and start the timer.
      iqs::mpi::StateBarrier();
      gettimeofday(&time, (struct timezone*)0);
      start = time.tv_sec + time.tv_usec * 1.0e-6;

      for (int iteration = 0; iteration < num_repetitions; iteration++) {
        if (control_qubit != target_qubit) {
          qubit_register.ApplyCPauliX(control_qubit, target_qubit);
        }
      }

      // MPI barrier and end the timer.
      iqs::mpi::StateBarrier();
      gettimeofday(&time, (struct timezone*)0);
      end = time.tv_sec + time.tv_usec * 1.0e-6;

      std::cout << "ComplexPosit, " << control_qubit << ", " << target_qubit << ", " << ((end - start) / double(num_repetitions)) << "\n";
    }
  }

/////////////////////////////////////////////////////////////////////////////////////////

  return 0;
}
