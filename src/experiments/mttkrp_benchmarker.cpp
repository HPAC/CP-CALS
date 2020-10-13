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

#include "experiments/mttkrp_benchmarker.h"

#include <vector>
#include <string>
#include <fstream>
#include <utils.h>

#include "cals_blas.h"
#include "timer.h"

using std::string;
using std::to_string;
using std::ofstream;
using std::cout;
using std::endl;

namespace cals::benchmarker
{
  void initialize_csv(string &file_name, const string &sep = ";")
  {
    auto file = ofstream(file_name, std::ios::out);
    file << "MODE" << sep
         << "TENSOR_MODES" << sep
         << "FLOPS" << sep
         << "MEMOPS" << sep
         << "RANK" << sep
         << "THREADS" << sep
         << "TIME" << endl;
  }

  void append_csv(string &file_name, int mode, string &modes_string, int rank, int threads,
                  uint64_t flops, uint64_t memops, double time, const string &sep = ";")
  {
    auto file = std::ofstream(file_name, std::ios::app);
    file << mode << sep
         << modes_string << sep
         << flops << sep
         << memops << sep
         << rank << sep
         << threads << sep
         << time << endl;
  }

  void benchmark_mttkrp(const vector<int> &modes, int num_threads, cals::ops::MttkrpParams &params)
  {
    Tensor T(modes);

    set_threads(num_threads);
    cout << "Threads: " << get_threads() << endl;

    ////////////////////////////////////////////////////////////////////////////////////
    //  Create CSV file name
    ////////////////////////////////////////////////////////////////////////////////////
    string folder = "../data/" + string(CALS_BACKEND) + "/benchmark/";

    string modes_string;
    for (auto m : modes) modes_string += to_string(m) + "-";
    modes_string.pop_back();

    string file_name;
    file_name = folder
        + "benchmark_"
        + string(CALS_BACKEND) + "_"
        + modes_string + "_"
        + to_string(get_threads()) + "_"
        + to_string(params.method)
        + ".csv";

    initialize_csv(file_name);
    cout << "File name: " << file_name << endl;

    ////////////////////////////////////////////////////////////////////////////////////
    //  Calculate target ranks
    ////////////////////////////////////////////////////////////////////////////////////
    vector<int> ranks;
    ranks.reserve(300);
    for (auto i = 1; i < 20; i += 1) ranks.push_back(i);
    for (auto i = 20; i < 100; i += 10) ranks.push_back(i);
    for (auto i = 100; i < 1000; i += 100) ranks.push_back(i);
    for (auto i = 1000; i <= 5000; i += 1000) ranks.push_back(i);

    for (auto m = 0; m < T.get_n_modes(); m++)
    {
      for (auto rank : ranks)
      {
        Timer gemm_t, krp_t, ts_gemm_t, ts_gemv_t;
        params.timer_mttkrp_gemm = &gemm_t;
        params.timer_mttkrp_krp = &krp_t;
        params.timer_twostep_gemm = &ts_gemm_t;
        params.timer_twostep_gemv = &ts_gemv_t;

        auto min_time = benchmark(rank, m, T, cals::ops::mttkrp, params);
        append_csv(file_name, m, modes_string, rank, num_threads, params.flops, params.memops, min_time);
      }
    }
  }
}
