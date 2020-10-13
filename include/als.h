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

#ifndef CALS_ALS_H
#define CALS_ALS_H

#include <fstream>
#include <iostream>

#include "utils.h"
#include "timer.h"
#include "tensor.h"
#include "ktensor.h"

namespace cals
{
  struct AlsReport
  {
    // Target tensor details
    int tensor_rank;
    int n_modes;
    vector<int> modes;
    double X_norm;

    // Execution parameters
    int iter = 0;
    int max_iter;
    int n_threads;
    int ktensor_id;
    int ktensor_rank;
    double tol;
    bool cuda;
    bool line_search;
    int line_search_interval;
    double line_search_step{0.9};
    update::UPDATE_METHOD update_method;

    // Execution data
    uint64_t flops_per_iteration;

    // Timers
    TIME(AlsTimer als_timer{};)
    TIME(vector<ModeTimer> mode_timers;)

    TIME(void init_timers()
         { mode_timers.resize(n_modes, ModeTimer()); })

    void print_header(const std::string &file_name, const std::string &sep = ";")
    {
      auto file = std::ofstream(file_name, std::ios::out);
      file << "TENSOR_RANK" << sep
           << "TENSOR_MODES" << sep
           << "KTENSOR_ID" << sep
           << "KTENSOR_RANK" << sep
           << "UPDATE_METHOD" << sep
           << "LINE_SEARCH" << sep
           << "MAX_ITERS" << sep
           << "ITER" << sep
           << "NUM_THREADS" << sep
           << "FLOPS" << sep;
               TIME( for (const auto &name : als_timer.names)
                 file << name << sep;
               for (auto i = 0lu; i < modes.size(); i++)
                 for (auto j : {ModeTimer::TOTAL_MTTKRP, ModeTimer::UPDATE})
                   file << "MODE_" << i << "_" << mode_timers[0].names[j] << sep;
      )
      file << std::endl;
    }

    void print_to_file(const std::string &file_name, const std::string &sep = ";")
    {
      auto file = std::ofstream(file_name, std::ios::app);

        file << tensor_rank << sep;
        for (auto m = modes.cbegin(); m < modes.cend() - 1; m++) file << *m << "-";
        file << *(modes.cend() - 1) << sep;
        file << ktensor_id << sep;
        file << ktensor_rank << sep;
        file << update::update_method_names[update_method] << sep;
        file << line_search << sep;
        file << max_iter << sep;
        file << iter << sep;
        file << n_threads << sep;
        file << flops_per_iteration << sep;
                 TIME(for (auto &timer : als_timer.timers)
                   file << timer.get_min() << sep;
                 for (auto &mode_timer : mode_timers)
                   for (auto j : {ModeTimer::TOTAL_MTTKRP, ModeTimer::UPDATE})
                     file << mode_timer.timers[j].get_min() << sep;
        )
        file << std::endl;
    }
  };

  struct AlsParams
  {
    double tol{1e-4};
    int max_iterations{150};
    ops::MTTKRP_METHOD mttkrp_method{ops::MTTKRP_METHOD::AUTO};
    int mttkrp_twostep_option{0};
    bool force_max_iter{false};
    update::UPDATE_METHOD update_method{update::UPDATE_METHOD::UNCONSTRAINED};
    bool cuda{false};
    bool line_search{false};
    int line_search_interval{5};
    double line_search_step{0.9};
  };

  AlsReport cp_als(const Tensor &X, Ktensor &ktensor, AlsParams &params);
}

#endif //CALS_ALS_H
