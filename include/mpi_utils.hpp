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

template<typename Posit>
using Bitblock = decltype(Posit().get());

template <typename Posit> struct ComplexBitblock {
  using bits = Bitblock<Posit>;

  ComplexBitblock(std::complex<Posit> p)
      : real(p.real().get()), imag(p.imag().get()) {}

  ComplexBitblock() = default;

  bits real;
  bits imag;

  std::complex<Posit> to_posit() const {
    Posit real_posit;
    real_posit.setBitblock(real);
    Posit imag_posit;
    imag_posit.setBitblock(imag);
    return std::complex(real_posit, imag_posit);
  }
};

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
std::vector<ComplexBitblock<IqsPosit24<es>>>
posit_buffer_to_byte_buffer(const ComplexPosit24<es> *posit_buffer,
                            const size_t posit_count) {

  std::vector<ComplexBitblock<IqsPosit24<es>>> byte_buffer;

  for(auto in = posit_buffer; in < posit_buffer + posit_count; in++) {
    auto& num = *in;
    byte_buffer.push_back(num);
  }

  return byte_buffer;
}

template <size_t es>
void byte_buffer_to_posit_buffer(const std::vector<ComplexBitblock<IqsPosit24<es>>> &byte_buffer, ComplexPosit24<es> *posit_buffer, size_t) {
  size_t i = 0;
  for(auto &bs : byte_buffer) {
    posit_buffer[i++] = bs.to_posit();
  }
}

template <typename Posit, typename F>
static void posit_reduce_bitblock(Bitblock<Posit> *in, Bitblock<Posit> *in_out, const F& func) {
  Posit in_posit = Posit().setBitblock((const Bitblock<Posit>)*in);
  Posit in_out_posit = Posit().setBitblock((const Bitblock<Posit>)*in_out);

  Posit result = func(in_posit, in_out_posit);  //,  > in_out_posit ? in_posit : in_out_posit;
  *in_out = result.get();
}

static void mpi_sum_posit24(void *in, void *in_out, int *len, MPI_Datatype *datatype) {
  if (*len != 1) {
    throw std::runtime_error("mpi_sum_posit24: len > 1 not implemented.");
  }

  auto reduce = [](const auto& l, const auto& r) {
    return l + r;
  };
  
  constexpr size_t nbits = 24;
  if (*datatype == mpi_datatype_handle_posit24_es0) {
    using Posit = sw::universal::posit<nbits, 0>;
    posit_reduce_bitblock<Posit>((Bitblock<Posit>*)in, (Bitblock<Posit>*)in_out, reduce);
  }
  else if (*datatype == mpi_datatype_handle_posit24_es1) {
    using Posit = sw::universal::posit<nbits, 1>;
    posit_reduce_bitblock<Posit>((Bitblock<Posit>*)in, (Bitblock<Posit>*)in_out, reduce);
  }
  else if (*datatype == mpi_datatype_handle_posit24_es2) {
    using Posit = sw::universal::posit<nbits, 2>;
    posit_reduce_bitblock<Posit>((Bitblock<Posit>*)in, (Bitblock<Posit>*)in_out, reduce);
  }
  else {
    std::cout << "Error: MPI datatype not supported: " << *datatype << std::endl;
    throw std::runtime_error("Invalid datatype for posit24 addition.");
  }
}

static void mpi_max_posit24(void *in, void *in_out, int *len, MPI_Datatype *datatype) {
  if (*len != 1) {
    throw std::runtime_error("mpi_max_posit24: len > 1 not implemented.");
  }

  auto reduce = [](const auto& l, const auto& r) {
    return max(l, r);
  };

  constexpr size_t nbits = 24;
  if (*datatype == mpi_datatype_handle_posit24_es0) {
    using Posit = sw::universal::posit<nbits, 0>;
    posit_reduce_bitblock<Posit>((Bitblock<Posit>*)in, (Bitblock<Posit>*)in_out, reduce);
  }
  else if (*datatype == mpi_datatype_handle_posit24_es1) {
    using Posit = sw::universal::posit<nbits, 1>;
    posit_reduce_bitblock<Posit>((Bitblock<Posit>*)in, (Bitblock<Posit>*)in_out, reduce);
  }
  else if (*datatype == mpi_datatype_handle_posit24_es2) {
    using Posit = sw::universal::posit<nbits, 2>;
    posit_reduce_bitblock<Posit>((Bitblock<Posit>*)in, (Bitblock<Posit>*)in_out, reduce);
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
  std::vector<ComplexBitblock<IqsPosit24<es>>> byte_recvbuf(recvcount);

  auto byte_sendbuf = posit_buffer_to_byte_buffer<es>(sendbuf, sendcount);

  auto scale = sizeof(ComplexBitblock<IqsPosit24<es>>);
  int ret_val = MPI_Sendrecv(byte_sendbuf.data(), sendcount * scale, MPI_BYTE, dest, sendtag,
                             byte_recvbuf.data(), recvcount * scale, MPI_BYTE, source, recvtag,
                             comm, status);
  
  /* std::transform(); */
  byte_buffer_to_posit_buffer<es>(byte_recvbuf, recvbuf, recvcount);

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

  auto byte_sendbuf = posit_buffer_to_byte_buffer<es>(data, 1);

  size_t scale = sizeof(ComplexBitblock<IqsPosit24<es>>);

  int ret_val = MPI_Bcast(byte_sendbuf.data(), scale, MPI_BYTE, root, comm);
  
  byte_buffer_to_posit_buffer<es>(byte_sendbuf, data, 1);

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
