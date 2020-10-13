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

#include "utils.h"

#include <stack>
#include <multi_ktensor.h>
#include <cmath>
#include <timer.h>
#include <fstream>

namespace cals::utils
{

  std::string mode_string(vector<int> const &modes)
  {
    std::string m_string;
    for (auto const &m : modes)
      m_string += std::to_string(m) + '-';
    m_string.pop_back();

    return m_string;
  }

  MttkrpLut read_lookup_table(vector<int> const &modes, int threads)
  {
    auto lut = MttkrpLut(modes.size());

    std::string blas_name = CALS_BACKEND;
    std::string dir =
        "../data/" + blas_name + "/lookup_tables/" + mode_string(modes) + "/" + std::to_string(threads) + "/";

    for (auto m = 0lu; m < modes.size(); m++)
    {
      auto file = std::ifstream(dir + std::to_string(m), std::ios::in);
      if (file.is_open())
      {
        int rank = 0, mttkrp_method = 0;

        while (file >> rank >> mttkrp_method)
          lut[m].insert(std::pair<int, int>(rank, mttkrp_method));
      } else
      {
        DEBUG(std::cout << "Lookup table missing for tensor: " << mode_string(modes) << ", Threads: " << threads << std::endl;);
        return MttkrpLut();
      }
    }
    return lut;
  }
}

namespace cals::ops
{
  struct Mttkrp2StepGemmParams
  {
    // Intermediate Matrix
    int inter_rows{};

    // DGEMM parameters
    CBLAS_TRANSPOSE trans_A{};
    int n_blocks{};
    int block_offset{};
    int block_rows{};
    int next_block{};
    int stride{};
    int B{};
  };

  struct Mttkrp2StepGemvParams
  {
    // DGEMV parameters
    CBLAS_TRANSPOSE trans_A{};
    int A_rows{};
    int A_cols{};
    int stride{};
    int x{};
    int y{};
  };

  void hadamard_all(vector<Matrix> &gramians)
  {
    for (auto i = 1lu; i < gramians.size(); i++) // skip first gramian
      gramians[0].hadamard(gramians[i]);
  }

  Matrix &hadamard_but_one(vector<Matrix> &gramians, int mode)
  {
    // Initialize target gramian
    gramians[mode].fill([]() -> double
                        { return static_cast<double>(1.0); });

    for (auto i = 0lu; i < gramians.size(); i++)
    {
      if (i == static_cast<unsigned long int>(mode)) // ...except mode...
        continue;
      else // ATA[n] := ATA[n] .* ATA[k]
        gramians[mode].hadamard(gramians[i]);
    }
    return gramians[mode];
  }

  Matrix &khatri_rao(const Matrix &A, const Matrix &B, Matrix &K, KrpParams &params)
  {
#if true
    {
      const int cols = K.get_cols();
      const int rowsA = A.get_rows();
      const int rowsB = B.get_rows();

#pragma omp parallel // NOLINT(openmp-use-default-none)
      {
        int index[2] = {0, 0};
#pragma omp for // NOLINT(openmp-use-default-none)
        for (int col = 0; col < cols; col++)
        {
          for (int row = 0; row < rowsA * rowsB; row++)
          {
            // Generate one column of the output.
            // Vector hadamard
            K.at(row, col) = A.at(index[0], col) * B.at(index[1], col);

            // Bump index.
            index[1]++;
            if (index[1] >= rowsB)
            {
              index[1] = 0;
//              cuda::send_to_device_async(K.get_data() + col * K.get_col_stride() + index[0] * rowsB,
//                                         K.get_cudata() + col * K.get_col_stride() + index[0] * rowsB,
//                                         rowsB, params.cuda_params.streams[index[0] % cuda_n_streams]);
              index[0]++;
            }
          }
          index[0] = 0;

          if (params.cuda)
          {
#if CUDA_ENABLED
            //            cuda::send_to_device_async(K.get_data() + col * K.get_col_stride(), K.get_cudata() + col * K.get_col_stride(),
            //                                       K.get_rows(), params.cuda_params.streams[0]);
#else
            std::cerr << "Not compiled with CUDA support" << std::endl;
            exit(EXIT_FAILURE);
#endif
          }
        }
      };

      if (params.cuda)
      {
#if CUDA_ENABLED
        //        cudaDeviceSynchronize();
#else
        std::cerr << "Not compiled with CUDA support" << std::endl;
        exit(EXIT_FAILURE);
#endif
      }

      params.flops += 1llu * K.get_cols() * K.get_rows();
      params.memops += 1llu * K.get_cols() * K.get_rows() +
                       1llu * A.get_rows() * A.get_cols() +
                       1llu * B.get_rows() * B.get_cols();
      return K;
    }

#else
    //#if CUDA_ENABLED
    //    {
    //      const int cols = K.get_cols();
    //      const int rowsA = A.get_rows();
    //      const int rowsB = B.get_rows();
    //
    //      auto handle = cuda::cublas::create_handle();
    //      for (int col = 0; col < cols; col++)
    //        for (int row = 0; row < rowsA; row++)
    //        {
    //          auto *K_pointer = K.get_cudata() + col * K.get_col_stride() + row * rowsB;
    //          cublasDcopy(handle, rowsB, B.get_cudata() + col * B.get_col_stride(), 1, K_pointer, 1);
    //          const double val = A.at(row,col);
    //          cublasDscal(handle, rowsB, &val, K_pointer, 1);
    //        }
    //      cuda::cublas::destroy_handle(handle);
    //    }
    //    return K;
    //#else
    //    {
    //      const int cols = K.get_cols();
    //      const int rowsA = A.get_rows();
    //      const int rowsB = B.get_rows();
    //
    //#pragma omp parallel for // NOLINT(openmp-use-default-none)
    //      for (int col = 0; col < cols; col++)
    //        for (int row = 0; row < rowsA; row++)
    //        {
    //          auto *K_pointer = K.get_data() + col * K.get_col_stride() + row * rowsB;
    //          cblas_dcopy(rowsB, B.get_data() + col * B.get_col_stride(), 1, K_pointer, 1);
    //          cblas_dscal(rowsB, A.at(row, col), K_pointer, 1);
    //        }
    //    }
    //#if CUDA_ENABLED
    //    K.send_to_device();
    //#endif
    //    return K;
    //#endif
#endif
  }

  // Internal recursive implementation of the Khatri-Rao product.
  Matrix &
  khatri_rao_rec(std::stack<Matrix *> &remaining_targets, vector<Matrix> &workspace, int w_index, KrpParams &params)
  {
    if (!remaining_targets.empty())
    {
      auto *factor = remaining_targets.top();
      remaining_targets.pop();
      auto &prev_target = workspace[w_index - 1];
      auto &new_target = workspace[w_index];

      new_target.resize(prev_target.get_rows() * factor->get_rows(), prev_target.get_cols());
      khatri_rao(prev_target, *factor, new_target, params);

      return khatri_rao_rec(remaining_targets, workspace, ++w_index, params);
    } else return workspace[w_index - 1];
  }

  // Compute ⨀ (n != mode) u.factor[n]
  Matrix &khatri_rao(Ktensor &u, vector<Matrix> &workspace, int mode, KrpParams &params)
  {
    params.flops = 0;
    params.memops = 0;

    // TODO see if you can preallocate size of stack
    std::stack<Matrix *> remaining_targets;
    for (auto i = 0; i < u.get_n_modes(); i++)
      if (i != mode) remaining_targets.push(&(u.get_factor(i)));

    int w_index = 0;
    auto *u0 = remaining_targets.top();
    remaining_targets.pop();
    auto *u1 = remaining_targets.top();
    remaining_targets.pop();

    workspace[w_index].resize(u0->get_rows() * u1->get_rows(), u0->get_cols());
    khatri_rao(*u0, *u1, workspace[w_index], params);

    return khatri_rao_rec(remaining_targets, workspace, ++w_index, params);
  }

  Matrix &mttkrp_impl(const Tensor &X, Ktensor &u, vector<Matrix> &workspace, int mode, MttkrpParams &params)
  {
    // Explicitly generate the Khatri-Rao product
    if (params.timer_mttkrp_krp != nullptr) params.timer_mttkrp_krp->start();

    auto &krp = khatri_rao(u, workspace, mode, params.krp_params);

    if (params.cuda)
    {
#if CUDA_ENABLED
      krp.send_to_device();
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
      exit(EXIT_FAILURE);
#endif
    }

    if (params.timer_mttkrp_krp != nullptr) params.timer_mttkrp_krp->stop();

    // Compute the matrix product
    if (params.timer_mttkrp_gemm != nullptr) params.timer_mttkrp_gemm->start();

    // Figure out how to implicitly unfold the tensor as one or more matrix blocks.
    Unfolding unfolding = X.implicit_unfold(mode);

    if (params.cuda)
    {
#if CUDA_ENABLED
      auto &handle = params.cuda_params.handle;
      auto const &one = params.cuda_params.one;
      auto const &zero = params.cuda_params.zero;

      cublasSetStream(handle, params.cuda_params.streams[0]);
      if (mode != 0)
      {
        auto &factor = u.get_factor(mode);
        cudaMemsetAsync(factor.get_cudata(), 0, factor.get_n_elements() * sizeof(double),
                        params.cuda_params.streams[0]);
      }
      for (auto block_idx = 0; block_idx < unfolding.n_blocks; block_idx++)
      {
        // Locate block of the mode-th unfolding of X.
        double const *X_mode_blk_ptr = X.get_cudata() + block_idx * unfolding.block_offset;
        int ldX_mode = unfolding.stride;

        // Locate block of krp.
        double const *krp_blk_ptr = krp.get_cudata() + block_idx * unfolding.cols;

        auto &G = u.get_factor(mode);

        if (mode == 0)
          cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, G.get_rows(), G.get_cols(), unfolding.cols,
                      &one, X_mode_blk_ptr, ldX_mode, krp_blk_ptr, krp.get_col_stride(),
                      &zero, G.get_cudata(), G.get_col_stride());
        else
          cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, G.get_rows(), G.get_cols(), unfolding.cols,
                      &one, X_mode_blk_ptr, ldX_mode, krp_blk_ptr, krp.get_col_stride(),
                      &one, G.get_cudata(), G.get_col_stride());
      }
      u.get_factor(mode).receive_from_device_async(params.cuda_params.streams[0]);
//    cudaDeviceSynchronize();
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
      exit(EXIT_FAILURE);
#endif
    } else
    {
      if (mode != 0) u.get_factor(mode).zero();
      for (auto block_idx = 0; block_idx < unfolding.n_blocks; block_idx++)
      {
        /**
         MODE 0
         ====================
         Compute
           G := G + X * K

         where
           - G and K are column-major
           - X is column-major

         No transpositions needed.

         MODES 1, 2, ...
         ====================
         Compute
           G := G + X * K  or rather  G := G + (X')' * K

         where
           - X is row-major
           - G, K are col-major

         We transpose X, since we are using col-major BLAS
         **/

        // Locate block of the mode-th unfolding of X.
        double const *X_mode_blk_ptr = X.get_data() + block_idx * unfolding.block_offset;
        int ldX_mode = unfolding.stride;

        // Locate block of krp.
        double const *krp_blk_ptr = krp.get_data() + block_idx * unfolding.cols;

        auto &G = u.get_factor(mode);

        if (mode == 0)
          cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                      G.get_rows(), G.get_cols(), unfolding.cols,
                      1.0, X_mode_blk_ptr, ldX_mode, krp_blk_ptr, krp.get_col_stride(),
                      0.0, G.get_data(), G.get_col_stride());
        else
          cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                      G.get_rows(), G.get_cols(), unfolding.cols,
                      1.0, X_mode_blk_ptr, ldX_mode, krp_blk_ptr, krp.get_col_stride(),
                      1.0, G.get_data(), G.get_col_stride());
      }
    }

    if (params.timer_mttkrp_gemm != nullptr) params.timer_mttkrp_gemm->stop();

    auto &G = u.get_factor(mode);
    // GEMM  --  blocks * (2 * m * n * k + 2 * m * n)
    params.flops = 1llu * unfolding.n_blocks *
                         (2llu * G.get_rows() * G.get_cols() * unfolding.cols +
                          2llu * G.get_rows() * G.get_cols());
    params.memops = 1llu * unfolding.n_blocks *
                         (1llu * G.get_rows() * unfolding.cols +
                          1llu * G.get_cols() * unfolding.cols +
                          1llu * G.get_rows() * G.get_cols());

    params.flops += params.krp_params.flops;
    params.memops += params.krp_params.memops;

    return u.get_factor(mode);
  }

  Matrix &mttkrp_twostep_impl(const Tensor &X, Ktensor &ktensor, Matrix &workspace,
                              Mttkrp2StepGemmParams &gemm_params, Mttkrp2StepGemvParams &gemv_params,
                              MttkrpParams &params)
  {
    if (params.timer_twostep_gemm != nullptr) params.timer_twostep_gemm->start();
    auto &B = ktensor.get_factor(gemm_params.B);
    // Perform TTM
    workspace.resize(gemm_params.inter_rows, ktensor.get_rank());

    if (params.cuda)
    {
#if CUDA_ENABLED
      auto const &handle = params.cuda_params.handle;
      auto const &one = params.cuda_params.one;
      auto const &zero = params.cuda_params.zero;

      cublasOperation_t opA;
      gemm_params.trans_A == CblasNoTrans ? opA = CUBLAS_OP_N : opA = CUBLAS_OP_T;

      for (auto blk = 0; blk < gemm_params.n_blocks; blk++)
      {
        cublasSetStream(handle, params.cuda_params.streams[blk % cuda_n_streams]);
        cublasDgemm(handle, opA, CUBLAS_OP_N,
                    gemm_params.block_rows, workspace.get_cols(), B.get_rows(), &one,
                    X.get_cudata() + blk * gemm_params.block_offset,
                    gemm_params.stride, // careful with the stride. Is it always this?
                    B.get_cudata(), B.get_col_stride(),
                    &zero, workspace.get_cudata() + blk * gemm_params.block_rows, workspace.get_col_stride());
      }
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
      exit(EXIT_FAILURE);
#endif
    } else
    {
#pragma omp parallel for // NOLINT(openmp-use-default-none)
      for (auto blk = 0; blk < gemm_params.n_blocks; blk++)
        cblas_dgemm(CblasColMajor, gemm_params.trans_A, CblasNoTrans,
                    gemm_params.block_rows, workspace.get_cols(), B.get_rows(), 1.0,
                    X.get_data() + blk * gemm_params.block_offset,
                    gemm_params.stride, // careful with the stride. Is it always this?
                    B.get_data(), B.get_col_stride(),
                    0.0, workspace.get_data() + blk * gemm_params.block_rows, workspace.get_col_stride());
    }

    if (params.timer_twostep_gemm != nullptr) params.timer_twostep_gemm->stop();


    // Perform series of TTVs
    if (params.timer_twostep_gemv != nullptr) params.timer_twostep_gemv->start();

    auto &x = ktensor.get_factor(gemv_params.x);
    auto &y = ktensor.get_factor(gemv_params.y);

    if (params.cuda)
    {
#if CUDA_ENABLED
      for (auto &stream : params.cuda_params.streams) cudaStreamSynchronize(stream);  // Make sure all DGEMMS are done

      auto const &handle = params.cuda_params.handle;
      auto const &one = params.cuda_params.one;
      auto const &zero = params.cuda_params.zero;

      cublasOperation_t opA;

      gemv_params.trans_A == CblasNoTrans ? opA = CUBLAS_OP_N : opA = CUBLAS_OP_T;
      for (auto col = 0; col < workspace.get_cols(); col++)
      {
        cublasSetStream(handle, params.cuda_params.streams[col % cuda_n_streams]);
        cublasDgemv(handle, opA, gemv_params.A_rows, gemv_params.A_cols, &one,
                    workspace.get_cudata() + col * workspace.get_col_stride(), gemv_params.stride,
                    x.get_cudata() + col * x.get_col_stride(), 1, &zero,
                    y.get_cudata() + col * y.get_col_stride(), 1);
//        y.receive_from_device_async(params.cuda_params.streams[col % cuda_n_streams]);
      }
      cudaDeviceSynchronize();
      y.receive_from_device();
#else
      std::cerr << "Not compiled with CUDA support" << std::endl;
      exit(EXIT_FAILURE);
#endif
    } else
    {
#pragma omp parallel for // NOLINT(openmp-use-default-none)
      for (auto col = 0; col < workspace.get_cols(); col++)
        cblas_dgemv(CblasColMajor, gemv_params.trans_A, gemv_params.A_rows, gemv_params.A_cols, 1.0,
                    workspace.get_data() + col * workspace.get_col_stride(), gemv_params.stride,
                    x.get_data() + col * x.get_col_stride(), 1, 0.0,
                    y.get_data() + col * y.get_col_stride(), 1);
    }

    if (params.timer_twostep_gemv != nullptr) params.timer_twostep_gemv->stop();


    // GEMM  --  blocks * (2 * m * n * k + 2 * m * n)
    params.flops = 1llu * gemm_params.n_blocks *
                      (2llu * gemm_params.block_rows * workspace.get_cols() * B.get_rows() +
                       2llu * gemm_params.block_rows * workspace.get_cols());
    params.memops = 1llu * gemm_params.n_blocks *
                      (1llu * gemm_params.block_rows * B.get_rows() +
                       1llu * B.get_rows() * workspace.get_cols() +
                       1llu * gemm_params.block_rows * workspace.get_cols());

    // GEMVs --  cols * (2 * m * n)
    params.flops += workspace.get_cols() * 2llu * gemv_params.A_rows * gemv_params.A_cols;
    params.memops += workspace.get_cols() * (1llu * gemv_params.A_rows * gemv_params.A_cols + 2llu * gemv_params.A_cols);

    return y;
  }

  Matrix &mttkrp_twostep(const Tensor &X, Ktensor &u, Matrix &workspace, int mode, MttkrpParams &params)
  {
    assert(u.get_n_modes() == 3);  // Twostep method only implemented for 3D tensors

    auto &U_0 = u.get_factor(0);
    auto &U_1 = u.get_factor(1);
    auto &U_2 = u.get_factor(2);

    Mttkrp2StepGemmParams gemm_params;
    Mttkrp2StepGemvParams gemv_params;

    if (mode == 2)
    {
      if (params.method == MTTKRP_METHOD::TWOSTEP0) // Option 0 (IK x J) * (J * R)
      {
        gemm_params.inter_rows = U_1.get_rows() * U_2.get_rows();
        gemm_params.trans_A = CblasTrans;
        gemm_params.stride = U_0.get_rows();
        gemm_params.block_rows = gemm_params.inter_rows;
        gemm_params.n_blocks = 1;
        gemm_params.B = 0;
        gemv_params.trans_A = CblasTrans;
        gemv_params.A_rows = U_1.get_rows();
        gemv_params.A_cols = U_2.get_rows();
        gemv_params.stride = U_1.get_rows();
        gemv_params.x = 1;
        gemv_params.y = 2;
      } else  // Option 1 (JK x I) * (I * R)
      {
        gemm_params.inter_rows = U_0.get_rows() * U_2.get_rows();
        gemm_params.trans_A = CblasNoTrans;
        gemm_params.block_offset = U_0.get_rows() * U_1.get_rows();
        gemm_params.stride = U_0.get_rows();
        gemm_params.block_rows = U_0.get_rows();
        gemm_params.n_blocks = U_2.get_rows();
        gemm_params.B = 1;

        gemv_params.trans_A = CblasTrans;
        gemv_params.A_rows = U_0.get_rows();
        gemv_params.A_cols = U_2.get_rows();
        gemv_params.stride = U_0.get_rows();
        gemv_params.x = 0;
        gemv_params.y = 2;
      }
    } else if (mode == 1)
    {
      if (params.method == MTTKRP_METHOD::TWOSTEP0)  // Option 0 (IJ x K) * (K * R)
      {
        gemm_params.inter_rows = U_0.get_rows() * U_1.get_rows();
        gemm_params.trans_A = CblasNoTrans;
        gemm_params.stride = U_0.get_rows() * U_1.get_rows();
        gemm_params.block_rows = gemm_params.inter_rows;
        gemm_params.n_blocks = 1;
        gemm_params.B = 2;
        gemv_params.trans_A = CblasTrans;
        gemv_params.A_rows = U_0.get_rows();
        gemv_params.A_cols = U_1.get_rows();
        gemv_params.stride = U_0.get_rows();
        gemv_params.x = 0;
        gemv_params.y = 1;
      } else  // Option 1 (JK x I) * (I * R)
      {
        gemm_params.inter_rows = U_1.get_rows() * U_2.get_rows();
        gemm_params.trans_A = CblasTrans;
        gemm_params.stride = U_0.get_rows();
        gemm_params.block_rows = gemm_params.inter_rows;
        gemm_params.n_blocks = 1;
        gemm_params.B = 0;
        gemv_params.trans_A = CblasNoTrans;
        gemv_params.A_rows = U_1.get_rows();
        gemv_params.A_cols = U_2.get_rows();
        gemv_params.stride = U_1.get_rows();
        gemv_params.x = 2;
        gemv_params.y = 1;
      }
    } else if (mode == 0)
    {
      if (params.method == MTTKRP_METHOD::TWOSTEP0) // Option 0 (IJ x K) * (K * R)
      {
        gemm_params.inter_rows = U_0.get_rows() * U_1.get_rows();
        gemm_params.trans_A = CblasNoTrans;
        gemm_params.stride = U_0.get_rows() * U_1.get_rows();
        gemm_params.block_rows = gemm_params.inter_rows;
        gemm_params.n_blocks = 1;
        gemm_params.B = 2;
        gemv_params.trans_A = CblasNoTrans;
        gemv_params.A_rows = U_0.get_rows();
        gemv_params.A_cols = U_1.get_rows();
        gemv_params.stride = U_0.get_rows();
        gemv_params.x = 1;
        gemv_params.y = 0;
      } else  // Option 1 (IK x J) * (J * R)
      {
        gemm_params.inter_rows = U_0.get_rows() * U_2.get_rows();
        gemm_params.trans_A = CblasNoTrans;
        gemm_params.block_offset = U_0.get_rows() * U_1.get_rows();
        gemm_params.stride = U_0.get_rows();
        gemm_params.block_rows = U_0.get_rows();
        gemm_params.n_blocks = U_2.get_rows();
        gemm_params.B = 1;

        gemv_params.trans_A = CblasNoTrans;
        gemv_params.A_rows = U_0.get_rows();
        gemv_params.A_cols = U_2.get_rows();
        gemv_params.stride = U_0.get_rows();
        gemv_params.x = 2;
        gemv_params.y = 0;
      }
    } else
    {
      std::cerr << "Illegal mode given (" << mode << ") for a " << X.get_n_modes() << "D tensor." << std::endl;
      abort();
    }

    return mttkrp_twostep_impl(X, u, workspace, gemm_params, gemv_params, params);
  }

  Matrix &mttkrp(const Tensor &X, Ktensor &u, vector<Matrix> &workspace, int mode, MttkrpParams &params)
  {
    if (X.get_n_modes() != 3)
      mttkrp_impl(X, u, workspace, mode, params);
    else
    {
      if (params.method == MTTKRP_METHOD::MTTKRP)
        mttkrp_impl(X, u, workspace, mode, params);
      else if ((params.method == MTTKRP_METHOD::TWOSTEP0)
                || (params.method == MTTKRP_METHOD::TWOSTEP1))
        mttkrp_twostep(X, u, workspace[0], mode, params);  // workspace[0] is size maxmode1*maxmode2 x rank
      else if (params.method == MTTKRP_METHOD::AUTO)
      {
        if (params.cuda)  // If CUDA, Twostep0 is the best alternative
        {
          params.method = MTTKRP_METHOD::TWOSTEP0;
          mttkrp_twostep(X, u, workspace[0], mode, params);
          params.method = MTTKRP_METHOD::AUTO;
        } else  // No CUDA
        {
          if (!params.lut.empty() && !params.lut_keys.empty())  // If LUT exists, read it
          {
            auto key = std::lower_bound(params.lut_keys.cbegin(), params.lut_keys.cend(), u.get_rank());
            if (key == params.lut_keys.end())
              key = params.lut_keys.end() - 1;
            auto val = params.lut[mode].at(*key);
            if (val == MTTKRP_METHOD::MTTKRP)
              mttkrp_impl(X, u, workspace, mode, params);
            else if (val == MTTKRP_METHOD::TWOSTEP0 || val == MTTKRP_METHOD::TWOSTEP1)
            {
              params.method = static_cast<MTTKRP_METHOD>(val);
              mttkrp_twostep(X, u, workspace[0], mode, params);
              params.method = MTTKRP_METHOD::AUTO;
            }
          } else  // If LUT does not exist, revert to common sense defaults
          {
            if (get_threads() != 1) // If multi-threaded, never Twostep0
            {
              if (mode == 1)
                mttkrp_impl(X, u, workspace, mode, params);
              else
              {
                params.method = MTTKRP_METHOD::TWOSTEP1;
                mttkrp_twostep(X, u, workspace[0], mode, params);  // workspace[0] is (maxmode1*maxmode2 x rank)
                params.method = MTTKRP_METHOD::AUTO;
              }
            } else  // If single threaded, always Twostep0
            {
              params.method = MTTKRP_METHOD::TWOSTEP0;
              mttkrp_twostep(X, u, workspace[0], mode, params);  // workspace[0] is (maxmode1*maxmode2 x rank)
              params.method = MTTKRP_METHOD::AUTO;
            }
          }
        }
//        if (std::max({X.get_mode(0), X.get_mode(1), X.get_mode(2)}) > 5 * X.get_mode(mode))
//          mttkrp_twostep(X, u, workspace[0], mode, params);
//        else
//          mttkrp_impl(X, u, workspace, mode, params);
      } else
      {
        std::cerr << "Invalid MTTKRP method selected." << std::endl;
        abort();
      }
    }
    return u.get_factor(mode);
  }

  double compute_fast_error(double X_norm, const vector<double> &lambda, const Matrix &last_factor,
                            const Matrix &last_mttkrp, const Matrix &gramian_hadamard)
  {
//    hadamard_all(gramians);

    // Sum the entries of the Hadamard product
    double term2 = 0.0;
    for (auto j = 0; j < gramian_hadamard.get_cols(); j++)
      for (auto i = 0; i < gramian_hadamard.get_rows(); i++)
        term2 += lambda[i] * lambda[j] * gramian_hadamard.at(i, j);

    // Compute the inner product of the last factor matrix and the last matrix G.
    double term3 = 0.0;
    for (auto j = 0; j < last_factor.get_cols(); j++)
      for (auto i = 0; i < last_factor.get_rows(); i++)
        term3 += lambda[j] * last_factor.at(i, j) * last_mttkrp.at(i, j);

    // Sum up the terms in the approximation error formula.
    double fast_error = 0.0;
    fast_error = std::fmax(X_norm * X_norm + term2 - 2 * term3, 0);
    fast_error = std::sqrt(fast_error);

    return fast_error;
  }

  void update_gramian(const Matrix &factor, Matrix &gramian)
  {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                gramian.get_rows(), gramian.get_cols(), factor.get_rows(), 1.0,
                factor.get_data(), factor.get_col_stride(),
                factor.get_data(), factor.get_col_stride(), 0.0,
                gramian.get_data(), gramian.get_col_stride());
  }

  double compute_error(const Tensor &X, Ktensor &ktensor, Matrix &krp_workspace, Matrix &ten_workspace)
  {
    KrpParams krp_params;
    const auto &C = ktensor.get_factor(2);
    const auto &B = ktensor.get_factor(1);
    const auto &A = ktensor.get_factor(0);

    ktensor.denormalize();

    krp_workspace.resize(B.get_rows() * C.get_rows(), B.get_cols());
    khatri_rao(C, B, krp_workspace, krp_params);

    ten_workspace.resize(A.get_rows(), krp_workspace.get_rows());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.get_rows(), krp_workspace.get_rows(), krp_workspace.get_cols(),
                1.0, A.get_data(), A.get_col_stride(), krp_workspace.get_data(), krp_workspace.get_col_stride(),
                0.0, ten_workspace.get_data(), ten_workspace.get_col_stride());

    for (auto i = 0; i < X.get_n_elements(); i++) ten_workspace[i] = X[i] - ten_workspace[i];

    ktensor.normalize();

    return ten_workspace.norm();
  }

  void line_search(Ktensor &ktensor, Ktensor &ls_ktensor, Matrix &krp_workspace, Matrix &ten_workspace,
                   vector<Matrix> &gramians, const Tensor &X, double X_norm, LineSearchParams &params)
  {
    for (auto f = 0; f < ls_ktensor.get_n_modes(); f++)
    {
      const auto &current = ktensor.get_factor(f);
      auto &old = ls_ktensor.get_factor(f);

      krp_workspace.resize(current.get_rows(), current.get_cols());
      for (auto i = 0; i < old.get_n_elements(); i++) krp_workspace[i] = current[i] - old[i];

      old.read(current);
      cblas_daxpy(old.get_n_elements(), params.step, krp_workspace.get_data(), 1, old.get_data(), 1);
    }
    for (auto i = 0lu; i < ktensor.get_lambda().size(); i++) ls_ktensor.get_lambda()[i] = ktensor.get_lambda()[i];

    double error = compute_error(X, ls_ktensor, krp_workspace, ten_workspace);

    const double old_error = ktensor.get_approx_error();
    const double old_fit = ktensor.get_fit();

    if (error < old_error)
    {
      DEBUG(std::cout << "Fast forwarding... id: " << ktensor.get_id(););
      DEBUG(std::cout << " Old error: " << old_error << std::endl;);
      for (auto f = 0; f < ls_ktensor.get_n_modes(); f++)
      {
        ktensor.get_factor(f).copy(ls_ktensor.get_factor(f));

        if (params.cuda)
        {
#if CUDA_ENABLED
          ktensor.get_factor(f).send_to_device_async(custream);
#else
          std::cerr << "Not compiled with CUDA support" << std::endl;
          exit(EXIT_FAILURE);
#endif
        }

        update_gramian(ktensor.get_factor(f), gramians[f]);
      }

      ktensor.set_approx_error(error);
      ktensor.set_approx_error_diff(std::fabs(error - old_error));

      const double fit = 1 - std::fabs(error) / X_norm;
      ktensor.set_fit(fit);
      ktensor.set_fit_diff(std::fabs(fit - old_fit));

      if (params.cuda)
      {
#if CUDA_ENABLED
        cudaDeviceSynchronize();
#else
        std::cerr << "Not compiled with CUDA support" << std::endl;
        exit(EXIT_FAILURE);
#endif
      }
    }
  }
}

namespace cals::update
{
  Matrix &calculate_sp(Matrix &ztx, Matrix &ztxp, Matrix &ztz, Matrix &ztzp, vector<bool> &R, int &info)
  {
    auto n_pset = std::count(R.cbegin(), R.cend(), false);
    ztxp.resize(n_pset, 1);
    ztzp.resize(n_pset, n_pset);

    auto index = 0;
    for (auto i = 0lu; i < R.size(); i++)
    {
      if (R[i] == false)
      {
        ztxp[index] = ztx[i];
        auto index2 = 0;
        for (auto j = 0lu; j < R.size(); j++)
          if (R[j] == false) ztzp.at(index, index2++) = ztz.at(i, j);
        index++;
      }
    }

    int oni = 1;
    int size = ztzp.get_rows();
    int lda = ztzp.get_col_stride();
    int ldb = ztxp.get_col_stride();
    dposv("L", &size, &oni, ztzp.get_data(), &lda, ztxp.get_data(), &ldb, &info);
//    if (info)
//      std::cerr << "als_update_factor: DPOSV returned info=" << info << std::endl;

    return ztxp;
  }

  Matrix &calculate_lagrangian_multipliers(Matrix &ztx, Matrix &ztz, Matrix &d, Matrix &w)
  {
    // TODO consider using DSYMV to potentially improve performance for larger (10?) matrices
    cblas_dgemv(CblasColMajor, CblasNoTrans, ztz.get_rows(), ztz.get_cols(),
                1.0, ztz.get_data(), ztz.get_rows(),
                d.get_data(), d.get_cols(), 0.0, w.get_data(), w.get_cols());
    for (auto i = 0; i < w.get_n_elements(); i++) w[i] = ztx[i] - w[i];
    return w;
  }

  // Update the factor matrix by applying non-negativity constraints to the solution.
  // factor is assumed to contain the result of the MTTKRP, while gramian contains the hadamard product of all gramians
  // (except for the gramian of the target factor matrix)
  Matrix &update_factor_non_negative_constrained(Matrix &factor, Matrix &gramian, vector<vector<bool>> &active_old)
  {
    // Create local buffers
    const double tol = 10 * 2.2204e-16 * gramian.one_norm() * gramian.get_cols();
    int info = 0;
    Matrix ztx(factor.get_cols(), 1);
    Matrix ztz(gramian.get_rows(), gramian.get_cols());
    Matrix d(factor.get_cols(), 1);
    Matrix w(factor.get_cols(), 1);
//     vector<bool> active(factor.get_cols(), true);

    Matrix s(factor.get_cols(), 1);
    Matrix ztzp(gramian.get_rows(), gramian.get_cols());
    Matrix ztxp(factor.get_cols(), 1);

    // Iterative algorithm applied to each row of the factor matrix
    for (int row = 0; row < factor.get_rows(); row++)
    {
      // Initialize local buffers
      d.zero();
      ztz.copy(gramian);
      // std::fill(active.begin(), active.end(), true);
      for (int i = 0; i < factor.get_cols(); i++) ztx[i] = factor[row + factor.get_rows() * i];

      // Load previous active set
      auto &active = active_old[row];

      // If all constraints are active, it's the first iteration, so no correction needed.
      if (std::find(active.begin(), active.end(), false) != active.end())
      {
        info = 0;
        auto &sp = calculate_sp(ztx, ztxp, ztz, ztzp, active, info);
        if (info)  // Cholesky failed, probably due to singularity. Start from d=0.
        {
          for (int i = 0; i < d.get_n_elements(); i++) active[i] = true;
          d.zero();
        } else
        {
          auto index = 0;
          for (auto i = 0lu; i < active.size(); i++) d[i] = active[i] ? 0.0 : sp[index++];

          // Modified inner loop (if a solution was negative, make it zero, update the active set, and retry)
          while (sp.min() < 0)
          {
            for (int i = 0; i < d.get_n_elements(); i++) if (d[i] < 0.0) d[i] = 0.0;
            for (int i = 0; i < d.get_n_elements(); i++) active[i] = (d[i] == 0.0);

            // If at any point we end up with a full active set or a bad cholesky, abort and start from d=0
            if (std::find(active.begin(), active.end(), false) == active.end() || info)
            {
              for (int i = 0; i < d.get_n_elements(); i++) active[i] = true;
              d.zero();
              break;
            }

            calculate_sp(ztx, ztxp, ztz, ztzp, active, info);

            if (info)  // Cholesky failed, probably due to singularity. Start from d=0.
            {
              for (int i = 0; i < d.get_n_elements(); i++) active[i] = true;
              d.zero();
              break;
            }

            index = 0;
            for (auto i = 0lu; i < active.size(); i++) d[i] = active[i] ? 0.0 : sp[index++];
          }
        }
      }

      // Begin Main loop
      w = calculate_lagrangian_multipliers(ztx, ztz, d, w);

      while (std::find(active.begin(), active.end(), true) != active.end() &&           // R ≠ ∅
             w.max(active) > tol)                                              //  max(wn) > tol , n in R
      {
        auto m = w.max_id(active);
        active[m] = false;

        auto &sp = calculate_sp(ztx, ztxp, ztz, ztzp, active, info);

        auto index = 0;
        for (auto i = 0lu; i < active.size(); i++) s[i] = active[i] ? 0.0 : sp[index++];

        // Inner Loop
        while (sp.min() <= 0)
        {
          auto a = DBL_MAX;
          for (int i = 0; i < d.get_n_elements(); i++)
            if (active[i] == false && s[i] <= tol)
            {
              auto tmp = d[i] / (d[i] - s[i]);
              if (tmp < a) a = tmp;
            }

          for (int i = 0; i < d.get_n_elements(); i++)
          {
            d[i] = d[i] + a * (s[i] - d[i]);
            assert(d[i] >= 0 || std::fabs(d[i]) < tol);
            if (std::fabs(d[i]) < tol && active[i] == false)
            {
              active[i] = true;
              d[i] = 0;
            }
          }

          calculate_sp(ztx, ztxp, ztz, ztzp, active, info);

          index = 0;
          for (auto i = 0lu; i < active.size(); i++) s[i] = active[i] ? 0.0 : sp[index++];
        }

        d.copy(s);
        w = calculate_lagrangian_multipliers(ztx, ztz, d, w);
      }
      for (int i = 0; i < factor.get_cols(); i++) factor[row + factor.get_rows() * i] = d[i];
    }
    return factor;
  }

  Matrix &update_factor_unconstrained(Matrix &factor, Matrix &gramian)
  {
    // Solve U_n * H = G for U_n.
    int info = 0;
    int size = gramian.get_rows();
    int lda = gramian.get_col_stride();
    dpotrf("L", &size, gramian.get_data(), &lda, &info);
    if (info)
      std::cerr << "als_update_factor: DPORTF returned info=" << info << std::endl;

    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                factor.get_rows(), factor.get_cols(), 1.0, gramian.get_data(), gramian.get_col_stride(),
                factor.get_data(), factor.get_col_stride());
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
                factor.get_rows(), factor.get_cols(), 1.0, gramian.get_data(), gramian.get_col_stride(),
                factor.get_data(), factor.get_col_stride());
    return factor;
  }
}
