#ifndef IQS_MPI_UTILS_HPP
#define IQS_MPI_UTILS_HPP

#ifdef INTELQS_HAS_MPI

/////////////////////////////////////////////////////////////////////////////////////////
// All methods involve MPI types in the arguments. Only available with MPI enabled.
// However there are two implementation depending on whether BIGMPI is used or not.
/////////////////////////////////////////////////////////////////////////////////////////

#include <complex>
#include <mpi.h>

#include "utils.hpp"

#ifdef BIGMPI
#include "bigmpi.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////

namespace iqs {

namespace mpi {

using Bitblock24 = sw::universal::internal::bitblock<24>;

static MPI_Datatype mpi_datatype_handle_complex_posit24;
const size_t bytes_per_complex_posit24 = 2 * 3;

/////////////////////////////////////////////////////////////////////////////////////////

#ifndef BIGMPI

template <size_t es>
void posit_buffer_to_byte_buffer(ComplexPosit24<es> *posit_buffer, std::vector<uint8_t> *byte_buffer, size_t posit_count) {
  const size_t nbytes = 3;

  for (size_t i = 0; i < posit_count; i++) {
    Bitblock24 real_bits = posit_buffer[i].real().get();
    Bitblock24 imag_bits = posit_buffer[i].imag().get();

    std::memcpy(&byte_buffer[(2*i) * nbytes], &real_bits, nbytes);
    std::memcpy(&byte_buffer[(2*i + 1) * nbytes], &imag_bits, nbytes);
  }
}

template <size_t es>
void byte_buffer_to_posit_buffer(std::vector<uint8_t> *byte_buffer, ComplexPosit24<es> *posit_buffer, size_t byte_count) {
  const size_t nbytes = 3;

  for (size_t i = 0; i < byte_count; i += 2 * nbytes) {
    Bitblock24 real_bits;
    Bitblock24 imag_bits;
    
    std::memcpy(&byte_buffer[i], &real_bits, nbytes);
    std::memcpy(&byte_buffer[i + nbytes], &imag_bits, nbytes);

    IqsPosit24<es> real = IqsPosit24<es>().setBitblock(real_bits);
    IqsPosit24<es> imag = IqsPosit24<es>().setBitblock(imag_bits);

    posit_buffer[i / (2 * nbytes)] = ComplexPosit24<es>(real, imag);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////
// Definitions without BigMPI
/////////////////////////////////////////////////////////////////////////////////////////

static int MPI_Allreduce_x(float *sendbuf, float *recvbuf, int count,
                           MPI_Op op, MPI_Comm comm)
{
  return MPI_Allreduce((void*)sendbuf, (void *)recvbuf, count, MPI_FLOAT, op, comm);
}

static int MPI_Allreduce_x(double *sendbuf, double *recvbuf, int count,
                           MPI_Op op, MPI_Comm comm)
{
  return MPI_Allreduce((void*)sendbuf, (void *)recvbuf, count, MPI_DOUBLE, op, comm);
}

template <size_t es>
static int MPI_Allreduce_x(IqsPosit24<es> *sendbuf, IqsPosit24<es> *recvbuf, int count, MPI_Op op, MPI_Comm comm)
{
  MPI_Op posit_op;

  if (op == MPI_SUM) {
    posit_op = MPI_SUM;
  }
  else if (op == MPI_MAX) {
    posit_op = MPI_MAX;
  }
  else {
    std::cout << "MPI_Allreduce_x: Unsupported operation for posits: " << op << std::endl;
    throw std::runtime_error("MPI_Allreduce_x: Unsupported MPI_Op");
  }

  return MPI_Allreduce((void*)sendbuf, (void *)recvbuf, count, mpi_datatype_handle_complex_posit24, op, comm);
}

/////////////////////////////////////////////////////////////////////////////////////////

static
int MPI_Sendrecv_x(ComplexSP *sendbuf, size_t sendcount, size_t dest, size_t sendtag,
                   ComplexSP *recvbuf, size_t recvcount, size_t source, size_t recvtag,
                   MPI_Comm comm, MPI_Status *status)
{
  return MPI_Sendrecv((void *)sendbuf, sendcount, MPI_CXX_FLOAT_COMPLEX, dest, sendtag,
                      (void *)recvbuf, recvcount, MPI_CXX_FLOAT_COMPLEX, source, recvtag,
                       comm, status);
}

static
int MPI_Sendrecv_x(ComplexDP *sendbuf, size_t sendcount, size_t dest, size_t sendtag,
                   ComplexDP *recvbuf, size_t recvcount, size_t source, size_t recvtag,
                   MPI_Comm comm, MPI_Status *status)
{
  return MPI_Sendrecv((void *)sendbuf, sendcount, MPI_CXX_DOUBLE_COMPLEX, dest, sendtag,
                      (void *)recvbuf, recvcount, MPI_CXX_DOUBLE_COMPLEX, source, recvtag,
                      comm, status);
}

template<size_t es>
static int MPI_Sendrecv_x(ComplexPosit24<es> *sendbuf, size_t sendcount, size_t dest, size_t sendtag,
                          ComplexPosit24<es> *recvbuf, size_t recvcount, size_t source, size_t recvtag,
                          MPI_Comm comm, MPI_Status *status)
{
  size_t byte_sendcount = sendcount * bytes_per_complex_posit24;
  size_t byte_recvcount = recvcount / bytes_per_complex_posit24;
  std::vector<uint8_t> byte_sendbuf(byte_sendcount);
  std::vector<uint8_t> byte_recvbuf(byte_recvcount);

  posit_buffer_to_byte_buffer<es>(sendbuf, &byte_sendbuf, sendcount);

  int ret_val = MPI_Sendrecv((void *)&byte_sendbuf, byte_sendcount, mpi_datatype_handle_complex_posit24, dest, sendtag,
                             (void *)&byte_recvbuf, byte_recvcount, mpi_datatype_handle_complex_posit24, source, recvtag,
                             comm, status);
  
  byte_buffer_to_posit_buffer<es>(&byte_recvbuf, recvbuf, recvcount);

  return ret_val;
}

/////////////////////////////////////////////////////////////////////////////////////////

static int MPI_Bcast_x(ComplexSP *data, int root, MPI_Comm comm)
{
  return MPI_Bcast((void*)data, 1, MPI_CXX_FLOAT_COMPLEX, root, comm);
}

static int MPI_Bcast_x(ComplexDP *data, int root, MPI_Comm comm)
{
  return MPI_Bcast((void*)data, 1, MPI_CXX_DOUBLE_COMPLEX, root, comm);
}

template<size_t es>
static int MPI_Bcast_x(ComplexPosit24<es> *data, int root, MPI_Comm comm)
{
  std::vector<uint8_t> byte_sendbuf(bytes_per_complex_posit24);

  posit_buffer_to_byte_buffer<es>(data, &byte_sendbuf, 1);

  int ret_val = MPI_Bcast((void*)&byte_sendbuf, 1, mpi_datatype_handle_complex_posit24, root, comm);
  
  byte_buffer_to_posit_buffer<es>(&byte_sendbuf, data, 1);

  return ret_val;
}

#else

/////////////////////////////////////////////////////////////////////////////////////////
// Definitions with BigMPI
/////////////////////////////////////////////////////////////////////////////////////////

static int MPI_Allreduce_x(float *sendbuf, float *recvbuf, int count,
                           MPI_Op op, MPI_Comm comm)
{
  return MPIX_Allreduce_x((void*)sendbuf, (void *)recvbuf, count, MPI_FLOAT, op, comm);
}

static int MPI_Allreduce_x(double *sendbuf, double *recvbuf, int count,
                           MPI_Op op, MPI_Comm comm)
{
  return MPIX_Allreduce_x((void*)sendbuf, (void *)recvbuf, count, MPI_DOUBLE, op, comm);
}

/////////////////////////////////////////////////////////////////////////////////////////

static
 int MPI_Sendrecv_x(ComplexSP *sendbuf, size_t sendcount, size_t dest, size_t sendtag,
                   ComplexSP *recvbuf, size_t recvcount, size_t source, size_t recvtag,
                   MPI_Comm comm, MPI_Status *status)
{
  return MPIX_Sendrecv_x((void *)sendbuf, sendcount, MPI_CXX_FLOAT_COMPLEX, dest, sendtag,
                         (void *)recvbuf, recvcount, MPI_CXX_FLOAT_COMPLEX, source, recvtag,
                         comm, status);
}

static
int MPI_Sendrecv_x(ComplexDP *sendbuf, size_t sendcount, size_t dest, size_t sendtag,
                   ComplexDP *recvbuf, size_t recvcount, size_t source, size_t recvtag,
                   MPI_Comm comm, MPI_Status *status)
{
  return MPIX_Sendrecv_x((void *)sendbuf, sendcount, MPI_CXX_DOUBLE_COMPLEX, dest, sendtag,
                         (void *)recvbuf, recvcount, MPI_CXX_DOUBLE_COMPLEX, source, recvtag,
                         comm, status);
}

/////////////////////////////////////////////////////////////////////////////////////////

static int MPI_Bcast_x(ComplexSP *data, int root, MPI_Comm comm)
{
 // Not sure of how it is defined with BigMPI.
 assert(0);
}

//

static int MPI_Bcast_x(ComplexDP *data, int root, MPI_Comm comm)
{
 // Not sure of how it is defined with BigMPI.
 assert(0);
}

#endif //BIGMPI

/////////////////////////////////////////////////////////////////////////////////////////

}	// end namespace mpi
}	// end namespace iqs

/////////////////////////////////////////////////////////////////////////////////////////

#endif	// INTELQS_HAS_MPI

/////////////////////////////////////////////////////////////////////////////////////////

#endif	// header guard IQS_MPI_UTILS_HPP
