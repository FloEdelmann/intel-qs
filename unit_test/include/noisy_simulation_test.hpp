#ifndef NOISY_SIMULATION_TEST_HPP
#define NOISY_SIMULATION_TEST_HPP

#ifdef INTELQS_HAS_MPI
#include <mpi.h>
#endif

#include "../../include/qureg.hpp"

//////////////////////////////////////////////////////////////////////////////
// Test fixture class.

class NoisySimulationTest : public ::testing::Test
{
 protected:

  NoisySimulationTest()
  {
#ifdef INTELQS_HAS_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks_);
    MPI_Comm_rank(MPI_COMM_WORLD, &pool_rank_id_);
#else
    num_ranks_ = 1;
    pool_rank_id_ = 0;
#endif

    // To ensure that each of two states has at least 2 amplitude per rank.
    int min_num_qubits = iqs::floor_power_of_two( num_ranks_) + 1;
    if (num_qubits_ < min_num_qubits)
        num_qubits_ = min_num_qubits;
  }

  // Just after the 'constructor'.
  void SetUp() override
  {
      // This kind of tests makes no sense without MPI and at least 2 ranks.
      if (num_ranks_<2)
          GTEST_SKIP();
  }

  // Just before the 'destructor'.
  void TearDown() override
  {
     if (iqs::mpi::Environment::GetNumStates() != 1)
         iqs::mpi::Environment::UpdateStateComm(1);
  }

  int num_qubits_= 6;
  float T1_ = 6.;
  float T2_ = 4.;
  float accepted_error_ = 1e-15;
  int pool_rank_id_;
  int num_ranks_;
};

//////////////////////////////////////////////////////////////////////////////
// Test macros:

TEST_F(NoisySimulationTest, OneStateAtATime)
{
  ASSERT_EQ( iqs::mpi::Environment::GetNumStates(), 1);
  // Currently pool=state, but not always pool=MPI_COMM_WORLD since pool is
  // defined only for the useful ranks.

  if (iqs::mpi::Environment::IsUsefulRank() == false)
      return;

  iqs::QubitRegister<ComplexDP> psi (num_qubits_,"base",1+8+16+32);
  // |psi> = |100111> = |"1+8+16+32">
  psi.ApplyHadamard(0);
  psi.ApplyHadamard(1);
  // |psi> = |-+0111>

  iqs::QubitRegister<ComplexDP> noisy_psi (psi);
  ASSERT_FLOAT_EQ( noisy_psi.ComputeOverlap(psi).real(), 1.);
  // Set the dissipation and decoherence times.
  noisy_psi.SetNoiseTimescales(T1_, T2_);
  // Noise gates require random numbers.
  std::size_t rng_seed = 7777;
  iqs::RandomNumberGenerator<float> rnd_generator;
  rnd_generator.SetSeedStreamPtrs(rng_seed);
  noisy_psi.SetRngPtr(&rnd_generator);
  // A certain time duration is spend with all qubits idle.
  float duration=5;
  for (int qubit=0; qubit<num_qubits_; ++qubit)
      noisy_psi.ApplyNoiseGate(qubit,duration);
  ASSERT_TRUE( noisy_psi.ComputeOverlap(psi).real() < 1.-accepted_error_);
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(NoisySimulationTest, TwoStates)
{
  ASSERT_EQ( iqs::mpi::Environment::GetNumStates(), 1);
  // Update state commutator.
  int num_states = 2;
  iqs::mpi::Environment::UpdateStateComm(num_states);
  ASSERT_EQ( num_states, iqs::mpi::Environment::GetNumStates() );
  if (iqs::mpi::Environment::IsUsefulRank() == false)
      return;

  int my_state_id = iqs::mpi::Environment::GetStateId();
  std::size_t index = my_state_id;
  iqs::QubitRegister<ComplexDP> psi (num_qubits_, "base", index);

  // The pool has two states, |0> and |1>.
  int qubit = 0;
  float incoherent_sum, probability;
  probability = psi.GetProbability(qubit);
  ASSERT_FLOAT_EQ( float(my_state_id), probability );
  // Sum up the probabilities incoherently.
  incoherent_sum
    = iqs::mpi::Environment::IncoherentSumOverAllStatesOfPool<float>(probability);
  ASSERT_FLOAT_EQ( incoherent_sum, 1. );

  // Now initialize all states of the pool to |"0">.
  psi.Initialize("base",0);
  iqs::QubitRegister<ComplexDP> noisy_psi (psi);
  // Noise gates require random numbers.
  std::size_t rng_seed = 7777;
  iqs::RandomNumberGenerator<float> rnd_generator;
  rnd_generator.SetSeedStreamPtrs(rng_seed);
  noisy_psi.SetRngPtr(&rnd_generator);
  // If purely dissipation, the population in state |0> should increase.
  noisy_psi.SetNoiseTimescales(T1_, T1_/2);
  float duration=T1_;
  for (int q=0; q<num_qubits_; ++q)
  {
      noisy_psi.ApplyNoiseGate(q,duration);
  }
  probability = noisy_psi.GetProbability(qubit);
  incoherent_sum
    = iqs::mpi::Environment::IncoherentSumOverAllStatesOfPool<float>(probability);
  float incoherent_average = incoherent_sum / float(num_states);
//iqs::mpi::PoolPrint("~~~~ prob : " + std::to_string(probability), true);//FIXME
//iqs::mpi::PoolPrint("~~~~ aver : " + std::to_string(incoherent_average), true);//FIXME
  ASSERT_GT( incoherent_average, accepted_error_ );
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(NoisySimulationTest, OneStatePerRank)
{
  ASSERT_EQ( iqs::mpi::Environment::GetNumStates(), 1);
  // Update state commutator.
  int num_states = num_ranks_;
  iqs::mpi::Environment::UpdateStateComm(num_states);
  ASSERT_EQ( iqs::mpi::Environment::GetStateRank(), 0 );
  ASSERT_EQ( iqs::mpi::Environment::GetStateSize(), 1 );
  ASSERT_EQ( iqs::mpi::Environment::GetPoolRank(), pool_rank_id_ );
  ASSERT_EQ( iqs::mpi::Environment::GetPoolSize(), num_states );

  // Initialize all states in different computational basis states (if possible).
  int my_state_id = iqs::mpi::Environment::GetStateId();
  ASSERT_EQ( my_state_id , pool_rank_id_);

  if ( iqs::mpi::Environment::IsUsefulRank() )
  {
      iqs::QubitRegister<ComplexDP> psi (num_qubits_,"base",0);
      ASSERT_EQ( psi.GlobalSize(), psi.LocalSize() );
      std::size_t index = my_state_id % psi.GlobalSize();
      psi.Initialize("base",index);
      // At this point: my_state_id=0 --> |0>
      //                my_state_id=k --> |k%2^n>
      int qubit = 0;
      float probability = psi.GetProbability(qubit);
      ASSERT_FLOAT_EQ(probability, float(index%2));
      float incoherent_sum
        = iqs::mpi::Environment::IncoherentSumOverAllStatesOfPool<float>(probability);
      if (num_states%2==0)
          ASSERT_FLOAT_EQ( incoherent_sum, float(num_states  )/2. );
      else
          ASSERT_FLOAT_EQ( incoherent_sum, float(num_states-1)/2. );
  }
  else
      ASSERT_TRUE(false);
}

//////////////////////////////////////////////////////////////////////////////

#endif	// header guard NOISY_SIMULATION_TEST_HPP
