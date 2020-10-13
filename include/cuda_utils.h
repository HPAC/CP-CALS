//@HEADER
// ******************************************************************************
//
//  CP-CALS: Software for computing the Canonical Polyadic Decomposition using
//  the Concurrent Alternating Least Squares Algorithm.
//
//  Copyright (c) 2020, Christos Psarras
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ******************************************************************************
//@HEADER

#ifndef CALS_CUDA_UTILS_H
#define CALS_CUDA_UTILS_H

#if CUDA_ENABLED

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

int const cuda_n_streams = 2;
static cudaStream_t custream{};

namespace cuda
{
  void identify_devices();

  cudaStream_t create_stream();

  void initialize_stream(cudaStream_t &stream);

  void destroy_stream(cudaStream_t &stream);

  inline void init_custream()
  { initialize_stream(custream); };

  inline void destroy_custream()
  { destroy_stream(custream); };

  void allocate(double* &d_A, long int size);

  void deallocate(double* &d_A);

  void allocate_async(double* &host_ptr, long int size);

  void deallocate_async(double* &host_ptr);

  void send_to_device(double* h_A, double* d_A, int size);

  void send_to_device_async(double* h_A, double* d_A, int size, cudaStream_t &stream);

  void receive_from_device(double* d_A, double* h_A, int size);

  void receive_from_device_async(double* d_A, double* h_A, int size, cudaStream_t &stream);
}

namespace cuda::cublas
{
  cublasHandle_t create_handle();

  void destroy_handle(cublasHandle_t &handle);
}

namespace cuda
{
  struct CudaParams
  {
    cublasHandle_t handle;
    cudaStream_t streams[cuda_n_streams]{};
    double const one{1.0};
    double const zero{0.0};

    CudaParams()
    {
      handle = cuda::cublas::create_handle();
      for (auto & stream : streams) initialize_stream(stream);
    }

    ~CudaParams()
    {
      cuda::cublas::destroy_handle(handle);
      for (auto & stream : streams) destroy_stream(stream);
    }
  };
}
#endif

#endif //CALS_CUDA_UTILS_H
