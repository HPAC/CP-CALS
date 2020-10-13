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

#ifndef CALS_MTTKRP_BENCHMARKER_H
#define CALS_MTTKRP_BENCHMARKER_H

#include <vector>
#include <ktensor.h>

#include "tensor.h"
#include "matrix.h"
#include "timer.h"
#include "utils.h"

const int iterations = 10;

namespace cals::benchmarker
{
  void benchmark_mttkrp(const std::vector<int> &modes, int num_threads, cals::ops::MttkrpParams &mttkrp_params);

  template<typename F>
  double benchmark(int rank, int mode, Tensor &T, F &&f, cals::ops::MttkrpParams &params)
  {
    // Initialize operands
    Matrix cache(2000, 2000);

    auto modes = T.get_modes();
    cals::Ktensor ktensor(rank, modes);

    std::sort(modes.begin(), modes.end(), std::greater<>());

    vector<Matrix> workspace;
    workspace.emplace_back(Matrix(modes[0] * modes[1], rank));

    vector<double> time;
    time.reserve(iterations);

    for (auto i = 0; i < iterations; i++)
    {
      T.randomize();
      workspace[0].randomize();
      ktensor.randomize();
      for(auto j = 0; j < cache.get_n_elements(); j++)cache[j] += 0.0001;

      std::forward<decltype(f)>(f)(T, ktensor, workspace, mode, params);
      time.push_back(params.timer_mttkrp_gemm->get_time() + params.timer_mttkrp_krp->get_time()
                     + params.timer_twostep_gemm->get_time() + params.timer_twostep_gemv->get_time());
    }
    return *std::min_element(time.begin(), time.end());
  }
}

#endif //CALS_MTTKRP_BENCHMARKER_H
