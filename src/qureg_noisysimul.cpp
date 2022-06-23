#include "../include/qureg.hpp"
#include <universal/math/math>

/// \addtogroup qureg
///  @{

/// @file qureg_noisysimul.cpp
/// @brief Define the @c QubitRegister methods related to implementing simulations
/// in presence of noise.

namespace iqs {

/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Set the timescale of dissipation and decoherence.
/// @param T1 dissipation time
/// @param T1 decoherence time

template <class Type>
void QubitRegister<Type>::SetNoiseTimescales (BaseType T1, BaseType T2)
{
  // Physically, dissipation also causes decoherence to enforce the positivity of the
  // density matrix. In particular, T2 >= T1/2;
  assert(T2 >= T1/2.);
  // Set internal variables of the Qubit Register.
  T_1_ = T1;
  T_2_ = T2;
  T_phi_ = 1./( 1./T2 - 1./(2.*T1) );
}

/////////////////////////////////////////////////////////////////////////////////////////
/// @brief Apply stochastic noise gate.
/// @param qubit the index of the involved qubit

template <class Type>
void QubitRegister<Type>::ApplyNoiseGate(unsigned qubit, BaseType duration)
{
  assert (rng_ptr_ != nullptr);
  if (duration==0) return;

  BaseType p_X , p_Y , p_Z ;
  p_X = (1. - sw::universal::exp(-duration/T_1_) )/4.;
  p_Y = (1. - sw::universal::exp(-duration/T_1_) )/4.;
  p_Z = (1. - sw::universal::exp(-duration/T_2_) )/2. + (1. - sw::universal::exp(-duration/T_1_) )/4.;
  assert( p_X>0 && p_Y>0 && p_Z>0 );

  // Computation of the standard deviations for the noise gate parameters
  BaseType s_X , s_Y , s_Z ;
  s_X = sw::universal::sqrt( -sw::universal::log(1.-p_X) );
  s_Y = sw::universal::sqrt( -sw::universal::log(1.-p_Y) );
  s_Z = sw::universal::sqrt( -sw::universal::log(1.-p_Z) );

  // Generate angle and direction of Pauli rotation for Pauli-twirl noise channel.
  // Each random number is shared between the ranks of the same quantum state.
  BaseType v_X , v_Y , v_Z;
  rng_ptr_->GaussianRandomNumbers(&v_X, 1, "state");
  v_X *= s_X /2.;
  rng_ptr_->GaussianRandomNumbers(&v_Y, 1, "state");
  v_Y *= s_Y /2.;
  rng_ptr_->GaussianRandomNumbers(&v_Z, 1, "state");
  v_Z *= s_Z /2.;

  // Direct construction of the 2x2 matrix corresponding to the noise gate
  //     U_noise = exp(-i v_X X) * exp(-i v_Y Y) * exp(-i v_Z Z)
  // Helpful quantities:
  //     (A) = cos v_z -i sin v_z
  //     (B) = cos v_x * cos v_Y -i sin v_X * sin v_Y
  //     (C) = cos v_x * sin v_Y -i sin v_X * cos v_Y
  // Then :
  //               | A*B   -A'*C' |
  //     U_noise = |              |
  //               | A*C    A'*B' |

  Type A , B , C ;
  A = { sw::universal::cos(v_Z) , -sw::universal::sin(v_Z) };
  B = { sw::universal::cos(v_X)*sw::universal::cos(v_Y) , -sw::universal::sin(v_X)*sw::universal::sin(v_Y) };
  C = { sw::universal::cos(v_X)*sw::universal::sin(v_Y) , -sw::universal::sin(v_X)*sw::universal::cos(v_Y) };

  iqs::TinyMatrix<Type, 2, 2, 32> U_noise;
  U_noise(0, 0) = A*B;
  U_noise(0, 1) = -sw::universal::conj(A)*sw::universal::conj(C);
  U_noise(1, 0) = A*C;
  U_noise(1, 1) =  sw::universal::conj(A)*sw::universal::conj(B);

  // Apply the noise gate
  QubitRegister<Type>::Apply1QubitGate(qubit,U_noise);
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

template class QubitRegister<ComplexSP>;
template class QubitRegister<ComplexDP>;
template class QubitRegister<ComplexPosit24es0>;
template class QubitRegister<ComplexPosit24es1>;
template class QubitRegister<ComplexPosit24es2>;

} // end namespace iqs

/// @}
