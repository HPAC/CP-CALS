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

#ifndef CALS_UTILS_H
#define CALS_UTILS_H

#include <string>
#include <sstream>
#include <map>
#include <tuple>

#include "ktensor.h"
#include "timer.h"

typedef vector<std::map<int, int>> MttkrpLut;

namespace cals::ops
{
  /* Enumeration of different MTTKRP methods.
   * */
  enum MTTKRP_METHOD
  {
    MTTKRP = 0,  /* Compute KRP explicitly and multiply with unfolded tensor. */
    TWOSTEP0,    /* Compute MTTKRP in two steps, involving one GEMM and multiple GEMVs. */
    TWOSTEP1,    /* Compute MTTKRP in two steps, involving multiple (parallelizable) GEMMs and multiple GEMVs. */
    AUTO         /* Let the Lookup Tables (if they exist) or our emprirical algorithm decide. */
  };

  struct LineSearchParams
  {
    bool cuda{false};  /* Indicate whether cuda is used (to stream results back to the GPU when done). */
    double step{0.0};  /* Factor with which to move on the line. */
  };

  struct KrpParams
  {
    bool cuda{false};    /* Indicate whether cuda is used (to stream results back to the GPU when done). */

    uint64_t flops{0};   /* Output: Number of FLOPs performed. */
    uint64_t memops{0};  /* Output: Number of MemOps performed. */

#if CUDA_ENABLED
    cuda::CudaParams cuda_params;
#endif
  };

  struct MttkrpParams
  {
    MTTKRP_METHOD method{AUTO};  /* Method to use to compute MTTKRP. */
    KrpParams krp_params{};      /* Parameters to be used for the computation of the KRP (if computed). */

    /* Optional timers to measure time in a fine grained manner. */
    Timer *timer_mttkrp_krp{nullptr};
    Timer *timer_mttkrp_gemm{nullptr};

    Timer *timer_twostep_gemm{nullptr};
    Timer *timer_twostep_gemv{nullptr};

    MttkrpLut lut{}; /* Lookup Table containing the "best" method to compute MTTKRP for the size of the tensor, rank of Ktensor and number of threads or GPU. */
    vector<int> lut_keys{}; /* Sorted keys of the Lookup Table, for searching. */

    u_int64_t flops{0};   /* Output: Number of FLOPs performed. */
    u_int64_t memops{0};  /* Output: Number of MemOps performed. */

    bool cuda{false};     /* Indicate whether cuda is used (to stream results back to the GPU when done). */

#if CUDA_ENABLED
    cuda::CudaParams cuda_params{};
#endif
  };

  /* Function to compute the MTTKRP.
   *
   * @param X Target Tensor.
   * @param u Ktensor to use for MTTKRP.
   * @param workspace Vector of Matrix of appropriate size, to be used as workspace.
   * @param mode Mode for which to compute the MTTKRP.
   * @param params Specific parameters for MTTKRP.
   *
   * @return Reference to the factor matrix of the /p mode factor Matrix of /p u.
   * */
  Matrix &mttkrp(const Tensor &X, Ktensor &u, vector<Matrix> &workspace, int mode, MttkrpParams &params);

  /* Compute the approximation error of the Ktensor based on the FastALS paper formula.
   *
   * @param X_norm Norm of the target tensor.
   * @param lambda Vector of lambdas of Ktensor.
   * @params last_factor Last Factor matrix of Ktensor.
   * @params last_mttkrp The resulting Matrix of the last MTTKRP performed in the current iteration.
   * @params gramian_hadamard Matrix containing the hadamard product of the gramians of the Ktensor.
   *
   * @return The error of the Ktensor based on the FastALS paper formula.
   * */
  double compute_fast_error(double X_norm, const vector<double> &lambda, const Matrix &last_factor,
                            const Matrix &last_mttkrp, const Matrix &gramian_hadamard);

  void update_gramian(const Matrix &factor, Matrix &gramian);

  //  Compute the hadamard product of all but the matrix in position "mode". Store the result in position "mode".
  Matrix &hadamard_but_one(vector<Matrix> &gramians, int mode);

  // Compute the hadamard product of all matrices and store the result in the first.
  void hadamard_all(vector<Matrix> &gramians);

  void line_search(Ktensor &ktensor, Ktensor &ls_ktensor, Matrix &krp_workspace, Matrix &ten_workspace,
                   vector<Matrix> &gramians, const Tensor &X, double X_norm, LineSearchParams &params);
}

namespace cals::update
{
  enum UPDATE_METHOD
  {
    UNCONSTRAINED = 0,
    NNLS,
    LENGTH
  };

  static const std::string update_method_names[UPDATE_METHOD::LENGTH] =
      {
          "unconstrained",
          "nnls"
      };

  // Update the factor matrix by solving the SPD system factor * gramian = G
  // hint: The factor matrix should initially hold the result of the MTTKRP
  Matrix &update_factor_unconstrained(Matrix &factor, Matrix &gramian);

  Matrix &update_factor_non_negative_constrained(Matrix &factor, Matrix &gramian, vector<vector<bool>> &active_old);
}

namespace cals::utils
{
  MttkrpLut read_lookup_table(vector<int> const &modes, int threads);

  template<typename T>
  inline void set_mttkrp_timers(T &rep, cals::ops::MttkrpParams &mttkrp_params, int mode)
  {
    mttkrp_params.timer_mttkrp_krp = &rep.mode_timers[mode].timers[cals::ModeTimer::MODE_TIMERS::MTTKRP_KRP];
    mttkrp_params.timer_mttkrp_gemm = &rep.mode_timers[mode].timers[cals::ModeTimer::MODE_TIMERS::MTTKRP_GEMM];

    mttkrp_params.timer_twostep_gemm = &rep.mode_timers[mode].timers[cals::ModeTimer::MODE_TIMERS::TWOSTEP_GEMM];
    mttkrp_params.timer_twostep_gemv = &rep.mode_timers[mode].timers[cals::ModeTimer::MODE_TIMERS::TWOSTEP_GEMV];
  }

  template<typename T>
  inline void set_mttkrp_timers(T &rep, cals::ops::MttkrpParams &mttkrp_params, int mode, int iter)
  {
    mttkrp_params.timer_mttkrp_krp = &rep.mode_timers[iter][mode].timers[cals::ModeTimer::MODE_TIMERS::MTTKRP_KRP];
    mttkrp_params.timer_mttkrp_gemm = &rep.mode_timers[iter][mode].timers[cals::ModeTimer::MODE_TIMERS::MTTKRP_GEMM];

    mttkrp_params.timer_twostep_gemm = &rep.mode_timers[iter][mode].timers[cals::ModeTimer::MODE_TIMERS::TWOSTEP_GEMM];
    mttkrp_params.timer_twostep_gemv = &rep.mode_timers[iter][mode].timers[cals::ModeTimer::MODE_TIMERS::TWOSTEP_GEMV];
  }
}


#endif //CALS_UTILS_H
