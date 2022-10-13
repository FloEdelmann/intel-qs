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

extern MPI_Datatype mpi_datatype_handle_posit24_es0;
extern MPI_Datatype mpi_datatype_handle_posit24_es1;
extern MPI_Datatype mpi_datatype_handle_posit24_es2;
extern MPI_Datatype mpi_datatype_handle_complex_posit24;
extern MPI_Op mpi_op_handle_sum_posit24;
extern MPI_Op mpi_op_handle_max_posit24;

constexpr size_t bytes_per_posit24 = 3;
constexpr size_t bytes_per_complex_posit24 = 2 * bytes_per_posit24;

/////////////////////////////////////////////////////////////////////////////////////////

#ifndef BIGMPI

template <size_t es>
void posit_buffer_to_byte_buffer(ComplexPosit24<es> *posit_buffer, std::vector<uint8_t> *byte_buffer, size_t posit_count) {
  for (size_t i = 0; i < posit_count; i++) {
    Bitblock24 real_bits = posit_buffer[i].real().get();
    Bitblock24 imag_bits = posit_buffer[i].imag().get();

    std::memcpy(&byte_buffer[(2*i) * bytes_per_posit24], &real_bits, bytes_per_posit24);
    std::memcpy(&byte_buffer[(2*i + 1) * bytes_per_posit24], &imag_bits, bytes_per_posit24);
  }
}

template <size_t es>
void byte_buffer_to_posit_buffer(std::vector<uint8_t> *byte_buffer, ComplexPosit24<es> *posit_buffer, size_t byte_count) {
  for (size_t i = 0; i < byte_count; i += 2 * bytes_per_posit24) {
    Bitblock24 real_bits;
    Bitblock24 imag_bits;
    
    std::memcpy(&byte_buffer[i], &real_bits, bytes_per_posit24);
    std::memcpy(&byte_buffer[i + bytes_per_posit24], &imag_bits, bytes_per_posit24);

    IqsPosit24<es> real = IqsPosit24<es>().setBitblock(real_bits);
    IqsPosit24<es> imag = IqsPosit24<es>().setBitblock(imag_bits);

    posit_buffer[i / bytes_per_complex_posit24] = ComplexPosit24<es>(real, imag);
  }
}

template <size_t es>
static void sum_posit24_bitblock(Bitblock24 *in, Bitblock24 *in_out) {
  IqsPosit24<es> in_posit = IqsPosit24<es>().setBitblock((const Bitblock24)*in);
  IqsPosit24<es> in_out_posit = IqsPosit24<es>().setBitblock((const Bitblock24)*in_out);

  IqsPosit24<es> sum = in_posit + in_out_posit;
  Bitblock24 sum_bits = sum.get();

  std::memcpy(in_out, &sum_bits, bytes_per_posit24);
}

static void mpi_sum_posit24(void *in, void *in_out, int *len, MPI_Datatype *datatype) {
  if (*len != 1) {
    throw std::runtime_error("mpi_sum_posit24: len > 1 not implemented.");
  }
  
  if (*datatype == mpi_datatype_handle_posit24_es0) {
    sum_posit24_bitblock<0>((Bitblock24*)in, (Bitblock24*)in_out);
  }
  else if (*datatype == mpi_datatype_handle_posit24_es1) {
    sum_posit24_bitblock<1>((Bitblock24*)in, (Bitblock24*)in_out);
  }
  else if (*datatype == mpi_datatype_handle_posit24_es2) {
    sum_posit24_bitblock<2>((Bitblock24*)in, (Bitblock24*)in_out);
  }
  else {
    std::cout << "Error: MPI datatype not supported: " << *datatype << std::endl;
    throw std::runtime_error("Invalid datatype for posit24 addition.");
  }
}

template <size_t es>
static void max_posit24_bitblock(Bitblock24 *in, Bitblock24 *in_out) {
  IqsPosit24<es> in_posit = IqsPosit24<es>().setBitblock((const Bitblock24)*in);
  IqsPosit24<es> in_out_posit = IqsPosit24<es>().setBitblock((const Bitblock24)*in_out);

  IqsPosit24<es> max = in_posit > in_out_posit ? in_posit : in_out_posit;
  Bitblock24 max_bits = max.get();

  std::memcpy(in_out, &max_bits, bytes_per_posit24);
}

static void mpi_max_posit24(void *in, void *in_out, int *len, MPI_Datatype *datatype) {
  if (*len != 1) {
    throw std::runtime_error("mpi_max_posit24: len > 1 not implemented.");
  }

  if (*datatype == mpi_datatype_handle_posit24_es0) {
    max_posit24_bitblock<0>((Bitblock24*)in, (Bitblock24*)in_out);
  }
  else if (*datatype == mpi_datatype_handle_posit24_es1) {
    max_posit24_bitblock<1>((Bitblock24*)in, (Bitblock24*)in_out);
  }
  else if (*datatype == mpi_datatype_handle_posit24_es2) {
    max_posit24_bitblock<2>((Bitblock24*)in, (Bitblock24*)in_out);
  }
  else {
    std::cout << "Error: MPI datatype not supported: " << *datatype << std::endl;
    throw std::runtime_error("Invalid datatype for posit24 max.");
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
  MPI_Datatype posit_datatype;

  if (op == MPI_SUM) {
    posit_op = mpi_op_handle_sum_posit24;
  }
  else if (op == MPI_MAX) {
    posit_op = mpi_op_handle_max_posit24;
  }
  else {
    std::cout << "MPI_Allreduce_x: Unsupported operation for posits: " << op << std::endl;
    throw std::runtime_error("MPI_Allreduce_x: Unsupported MPI_Op");
  }

  if (es == 0) {
    posit_datatype = mpi_datatype_handle_posit24_es0;
  }
  else if (es == 1) {
    posit_datatype = mpi_datatype_handle_posit24_es1;
  }
  else if (es == 2) {
    posit_datatype = mpi_datatype_handle_posit24_es2;
  }
  else {
    std::cout << "MPI_Allreduce_x: Unsupported es for posits: " << es << std::endl;
    throw std::runtime_error("MPI_Allreduce_x: Unsupported es");
  }

  return MPI_Allreduce((void*)sendbuf, (void *)recvbuf, count, posit_datatype, posit_op, comm);
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
